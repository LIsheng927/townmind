using System;
using System.Collections.Generic;
using TownMind.Net;
using UnityEngine;

namespace TownMind.World
{
    /// <summary>
    /// 单个 NPC 的"身体"：只负责执行动作和上报观察，不做任何决策。
    /// 决策全部来自服务端（"大脑"）。
    /// </summary>
    public class NpcController : MonoBehaviour
    {
        public string NpcId;
        public float Speed = 2.5f;

        private TownClient _client;
        private Vector3? _target;
        private bool _awaitingAction;
        private float _nextReportTime;
        private float _requestSentTime;
        private const float ReplyTimeout = 12f; // 要大于服务端 LLM 超时(8s)
        private TextMesh _speech;
        private float _speechExpire;

        public void Init(string id, TownClient client)
        {
            NpcId = id;
            _client = client;
            _client.OnMessage += HandleMessage;
        }

        private void OnDestroy()
        {
            if (_client != null) _client.OnMessage -= HandleMessage;
        }

        private void Update()
        {
            if (_target.HasValue)
            {
                transform.position = Vector3.MoveTowards(transform.position, _target.Value, Speed * Time.deltaTime);
                if ((transform.position - _target.Value).sqrMagnitude < 0.01f) _target = null;
                return;
            }

            // 等回复超时：放弃这次请求，允许重发（服务端丢消息/慢时 NPC 不会卡死）
            if (_awaitingAction && Time.time - _requestSentTime > ReplyTimeout)
            {
                Debug.LogWarning($"[{NpcId}] reply timeout, retrying");
                _awaitingAction = false;
            }

            // 空闲 且 没有在等回复 且 已连上服务端 -> 上报观察，请求下一步指令
            if (!_awaitingAction && _client.IsConnected && Time.time >= _nextReportTime)
            {
                _awaitingAction = true;
                _requestSentTime = Time.time;
                var p = transform.position;
                SendObservation(p);
            }
        }

        private void ShowSpeech(string text)
        {
            if (_speech != null) Destroy(_speech.gameObject);
            var go = new GameObject("Speech");
            go.transform.SetParent(transform, false);
            go.transform.localPosition = new Vector3(0, 1.2f, 0);
            _speech = go.AddComponent<TextMesh>();
            _speech.text = text;
            _speech.characterSize = 0.12f;
            _speech.fontSize = 48;
            _speech.anchor = TextAnchor.LowerCenter;
            _speech.alignment = TextAlignment.Center;
            _speech.color = Color.white;
            // Unity 内置字体不含中文，改用系统字体（Windows 上是微软雅黑）
            var font = Font.CreateDynamicFontFromOSFont(new[] { "Microsoft YaHei", "SimHei", "Arial" }, 48);
            _speech.font = font;
            go.GetComponent<MeshRenderer>().material = font.material;
            _speechExpire = Time.time + 4f;
        }

        private void LateUpdate()
        {
            if (_speech == null) return;
            if (Time.time > _speechExpire) { Destroy(_speech.gameObject); return; }
            if (Camera.main != null) _speech.transform.rotation = Camera.main.transform.rotation; // 始终朝向相机
        }

        private async void SendObservation(Vector3 p)
        {
            var ok = await _client.Send(new Envelope
            {
                Type = "observation",
                NpcId = NpcId,
                Payload = new Dictionary<string, object> { { "pos", new[] { p.x, p.z } } }
            });
            if (!ok) _awaitingAction = false; // 没发出去，下一帧重试
        }

        private void HandleMessage(Envelope msg)
        {
            if (msg.Type != "action" || msg.NpcId != NpcId) return;
            _awaitingAction = false;
            _nextReportTime = Time.time + 1f; // 到达后停 1 秒再要新指令

            var name = Convert.ToString(msg.Payload["name"]);
            switch (name)
            {
                case "move_to":
                    var x = Convert.ToSingle(msg.Payload["x"]);
                    var z = Convert.ToSingle(msg.Payload["z"]);
                    _target = new Vector3(x, transform.position.y, z);
                    break;
                case "say":
                    ShowSpeech(Convert.ToString(msg.Payload["text"]));
                    _nextReportTime = Time.time + 3f; // 说话期间先别急着要下一步
                    break;
                case "idle":
                    _nextReportTime = Time.time + Convert.ToSingle(msg.Payload["seconds"]);
                    break;
                default:
                    Debug.LogWarning($"[{NpcId}] unknown action {name}");
                    break;
            }
        }
    }
}
