using System.Collections.Generic;
using Newtonsoft.Json;

namespace TownMind.Net
{
    /// <summary>与服务端 server/townmind/protocol.py 中的 Envelope 一一对应。</summary>
    public class Envelope
    {
        [JsonProperty("type")] public string Type;
        [JsonProperty("npc_id", NullValueHandling = NullValueHandling.Ignore)] public string NpcId;
        [JsonProperty("payload")] public Dictionary<string, object> Payload = new Dictionary<string, object>();
    }
}
