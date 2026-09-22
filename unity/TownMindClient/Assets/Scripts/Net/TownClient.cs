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

        public bool IsConnected => _ws != null && _ws.State == WebSocketState.Open;

        private ClientWebSocket _ws;
        private CancellationTokenSource _cts;
        private readonly ConcurrentQueue<Envelope> _inbox = new ConcurrentQueue<Envelope>();

        private const float ReconnectDelaySeconds = 5f; // 跟网页版 demo（web_demo/index.html）的重连间隔保持一致

        private async void Start()
        {
            _cts = new CancellationTokenSource();
            await ConnectLoop(_cts.Token);
        }

        /// <summary>连不上、或者连上了又断开，都在这里重试，不会因为"Unity 先于服务端启动"
        /// 或者"服务端中途重启"就再也连不上了——之前的写法只在 Start() 里试一次，失败就
        /// 彻底放弃，跟网页版 demo（onclose 里 5 秒后自动重连）不是同一个健壮程度。</summary>
        private async Task ConnectLoop(CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    _ws = new ClientWebSocket();
                    await _ws.ConnectAsync(new Uri(serverUrl), ct);
                    Debug.Log($"[TownClient] connected to {serverUrl}");
                    await Send(new Envelope { Type = "hello" });
                    await ReceiveLoop(ct); // 阻塞到断开为止，断开后走到下面重试
                    Debug.LogWarning("[TownClient] disconnected, retrying...");
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[TownClient] connect failed: {e.Message}，{ReconnectDelaySeconds:0}秒后重试"
                        + "（服务端启动了吗？cd server && uv run uvicorn townmind.main:app --port 8000）");
                }
                if (ct.IsCancellationRequested) break;
                try { await Task.Delay(TimeSpan.FromSeconds(ReconnectDelaySeconds), ct); }
                catch (OperationCanceledException) { break; }
            }
        }

        /// <summary>发送成功返回 true；未连接或发送失败返回 false（调用方据此决定是否重试）。</summary>
        public async Task<bool> Send(Envelope msg)
        {
            if (!IsConnected) return false;
            try
            {
                var bytes = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(msg));
                await _ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, _cts.Token);
                return true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[TownClient] send failed: {e.Message}");
                return false;
            }
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
