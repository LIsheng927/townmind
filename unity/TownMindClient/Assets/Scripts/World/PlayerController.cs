using System.Collections.Generic;
using TownMind.Net;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace TownMind.World
{
    /// <summary>
    /// 玩家的"身体"：WASD/方向键移动，周期性上报位置，输入框里打字回车对附近的 NPC 说话。
    /// 跟 NpcController 共用同一套协议（position / player_say），服务端早就测过（tests/test_ws.py）、
    /// 网页版 demo（server/web_demo/index.html）也验证过能跑通——这里只是把同一条链路接进
    /// Unity 客户端本身，不是重新设计一遍协议。
    /// </summary>
    public class PlayerController : MonoBehaviour
    {
        public float Speed = 3.5f;
        private const float WorldHalf = 8f; // 跟 townmind/policy.py 的 WORLD_HALF_SIZE 对齐
        private const float PositionInterval = 0.5f; // 跟 NpcController 的上报间隔保持一致

        private TownClient _client;
        private InputField _input;
        private float _nextPositionTime;

        public void Init(TownClient client, InputField input)
        {
            _client = client;
            _input = input;
        }

        private void Update()
        {
            bool typing = _input != null && EventSystem.current != null
                          && EventSystem.current.currentSelectedGameObject == _input.gameObject;

            if (typing)
            {
                // 自己判断回车，不依赖 InputField 自带的 onSubmit/onEndEdit——那两个在"点别处失焦"
                // 和"按回车"之间不好区分，容易误触发；跟网页版 demo 里 keydown 判断 Enter 是同一个思路。
                if (Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.KeypadEnter))
                    SubmitText();
            }
            else
            {
                float dx = 0f, dz = 0f;
                if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow)) dz += 1f;
                if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow)) dz -= 1f;
                if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow)) dx -= 1f;
                if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow)) dx += 1f;
                if (dx != 0f || dz != 0f)
                {
                    var dir = new Vector3(dx, 0, dz).normalized;
                    var next = transform.position + dir * Speed * Time.deltaTime;
                    next.x = Mathf.Clamp(next.x, -WorldHalf, WorldHalf);
                    next.z = Mathf.Clamp(next.z, -WorldHalf, WorldHalf);
                    transform.position = next;
                }
                if (Input.GetKeyDown(KeyCode.Return) && _input != null)
                {
                    _input.Select();
                    _input.ActivateInputField();
                }
            }

            if (_client != null && _client.IsConnected && Time.time >= _nextPositionTime)
            {
                _nextPositionTime = Time.time + PositionInterval;
                SendPosition();
            }
        }

        public void SubmitText()
        {
            if (_input == null) return;
            var text = _input.text.Trim();
            _input.text = "";
            if (string.IsNullOrEmpty(text) || _client == null) return;
            var p = transform.position;
            // TownClient.Update() 只打印收到的消息（"<-"），发出去的东西之前完全没有日志——
            // 打了字按回车却在 Console 里看不到任何反应，不知道是没发出去还是发了没人回，
            // 这里补一条，至少能确认"确实发出去了、发的是这句话"。
            Debug.Log($"[Player] -> player_say: {text}");
            _ = _client.Send(new Envelope
            {
                Type = "player_say",
                Payload = new Dictionary<string, object>
                {
                    { "text", text },
                    { "pos", new[] { p.x, p.z } },
                },
            });
        }

        private async void SendPosition()
        {
            var p = transform.position;
            await _client.Send(new Envelope
            {
                Type = "position",
                NpcId = "player",
                Payload = new Dictionary<string, object> { { "pos", new[] { p.x, p.z } } },
            });
        }
    }
}
