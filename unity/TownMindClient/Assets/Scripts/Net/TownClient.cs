using System;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;

namespace TownMind.Net
{
    /// <summary>
    /// 连接 Python Agent 服务端的 WebSocket 客户端。
    /// 网络收发在后台线程，收到的消息放进队列，由 Update() 在主线程分发
    /// （Unity 的 API 只能在主线程调用）。
    /// </summary>
    public class TownClient : MonoBehaviour
    {
        [SerializeField] private string serverUrl = "ws://127.0.0.1:8000/ws";

        public event Action<Envelope> OnMessage;

        private ClientWebSocket _ws;
        private CancellationTokenSource _cts;
        private readonly ConcurrentQueue<Envelope> _inbox = new ConcurrentQueue<Envelope>();

        private async void Start()
        {
            _cts = new CancellationTokenSource();
            try
            {
                _ws = new ClientWebSocket();
                await _ws.ConnectAsync(new Uri(serverUrl), _cts.Token);
                Debug.Log($"[TownClient] connected to {serverUrl}");
                _ = ReceiveLoop(_cts.Token);
                await Send(new Envelope { Type = "hello" });
            }
            catch (Exception e)
            {
                Debug.LogError($"[TownClient] connect failed: {e.Message}");
            }
        }

        public async Task Send(Envelope msg)
        {
            if (_ws == null || _ws.State != WebSocketState.Open) return;
            var bytes = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(msg));
            await _ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, _cts.Token);
        }

        private async Task ReceiveLoop(CancellationToken ct)
        {
            var buffer = new byte[16 * 1024];
            var sb = new StringBuilder();
            try
            {
                while (!ct.IsCancellationRequested && _ws.State == WebSocketState.Open)
                {
                    var result = await _ws.ReceiveAsync(new ArraySegment<byte>(buffer), ct);
                    if (result.MessageType == WebSocketMessageType.Close) break;
                    sb.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                    if (!result.EndOfMessage) continue; // 一条消息可能分多帧到达
                    var msg = JsonConvert.DeserializeObject<Envelope>(sb.ToString());
                    sb.Clear();
                    if (msg != null) _inbox.Enqueue(msg);
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception e)
            {
                Debug.LogError($"[TownClient] receive loop error: {e.Message}");
            }
        }

        private void Update()
        {
            while (_inbox.TryDequeue(out var msg))
            {
                Debug.Log($"[TownClient] <- {msg.Type} {JsonConvert.SerializeObject(msg.Payload)}");
                OnMessage?.Invoke(msg);
            }
        }

        private void OnDestroy()
        {
            _cts?.Cancel();
            _ws?.Dispose();
        }
    }
}
