using Newtonsoft.Json.Linq;
using TownMind.Net;
using UnityEngine;

namespace TownMind.World
{
    /// <summary>
    /// 用代码搭建小镇：地面 + 10 个方块 NPC + 俯视相机；
    /// 建筑（面包店、铁匠铺、广场、磨坊、酒馆、杂货铺）由服务端在 welcome 消息里下发，收到后再盖。
    /// 这样"小镇有哪些地点"只在服务端的 world.py 里维护一份，不会两边不一致。
    /// 挂在和 TownClient 相同的物体上。
    /// </summary>
    [RequireComponent(typeof(TownClient))]
    public class TownWorld : MonoBehaviour
    {
        private TownClient _client;
        private bool _built;

        private void Start()
        {
            _client = GetComponent<TownClient>();
            _client.OnMessage += OnMessage;

            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.localScale = new Vector3(2f, 1f, 2f); // Plane 默认 10x10，这里变成 20x20
            ground.GetComponent<Renderer>().material.color = new Color(0.35f, 0.5f, 0.3f);

            // 出生点和服务端 evals/sim.py 里的 START_POSITIONS 保持一致
            Spawn("alice", new Vector3(-3, 0.5f, 0), Color.red);
            Spawn("bob", new Vector3(0, 0.5f, 3), Color.blue);
            Spawn("carol", new Vector3(3, 0.5f, -2), Color.yellow);
            Spawn("dan", new Vector3(5, 0.5f, 0), new Color(0.6f, 0.3f, 0.1f));
            Spawn("elsa", new Vector3(-5, 0.5f, 0), new Color(0.9f, 0.6f, 0.2f));
            Spawn("finn", new Vector3(-1.5f, 0.5f, 4.5f), Color.magenta);
            Spawn("greta", new Vector3(2, 0.5f, 2), Color.cyan);
            Spawn("iris", new Vector3(-4, 0.5f, 2), new Color(1f, 0.75f, 0.8f));
            Spawn("jonas", new Vector3(1, 0.5f, -2), Color.gray);
            Spawn("milo", new Vector3(-1, 0.5f, -1), Color.green);
            SpawnPlayer(new Vector3(0, 0.5f, -5));

            var cam = Camera.main;
            if (cam != null)
            {
                cam.transform.position = new Vector3(0, 16, -12);
                cam.transform.rotation = Quaternion.Euler(55, 0, 0);
            }
        }

        private void OnDestroy()
        {
            if (_client != null) _client.OnMessage -= OnMessage;
        }

        private void Spawn(string id, Vector3 pos, Color color)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = id;
            go.transform.position = pos;
            go.GetComponent<Renderer>().material.color = color;
            go.AddComponent<NpcController>().Init(id, _client);
        }

        /// <summary>玩家的身体：胶囊体 + 底部聊天输入框，都是代码搭的，不用在编辑器里手动拖场景。
        /// 跟 NPC 共用同一个 TownClient 连接（同一条 WebSocket，服务端按 npc_id/position 区分谁是谁）。</summary>
        private void SpawnPlayer(Vector3 pos)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            go.name = "player";
            go.transform.position = pos;
            go.GetComponent<Renderer>().material.color = new Color(0.16f, 0.37f, 0.66f); // 蓝色，跟网页版 demo 里玩家的颜色一致
            var label = TextLabel.Make(go.transform, "你", new Vector3(0, 1.3f, 0), 0.18f, Color.white);
            if (Camera.main != null) label.transform.rotation = Camera.main.transform.rotation;

            var input = PlayerHud.Create();
            var controller = go.AddComponent<PlayerController>();
            controller.Init(_client, input);
        }

        private void OnMessage(Envelope msg)
        {
            if (msg.Type != "welcome" || _built) return;
            if (!msg.Payload.TryGetValue("locations", out var raw) || !(raw is JArray locations)) return;
            _built = true;
            foreach (var item in locations) BuildLocation((JObject)item);
        }

        private static void BuildLocation(JObject o)
        {
            var name = (string)o["name"];
            var x = (float)o["x"];
            var z = (float)o["z"];
            var kind = (string)o["kind"];

            GameObject go;
            float labelHeight;
            if (kind == "plaza")
            {
                go = GameObject.CreatePrimitive(PrimitiveType.Cylinder); // 一块扁平的圆形广场
                go.transform.position = new Vector3(x, 0.05f, z);
                go.transform.localScale = new Vector3(7f, 0.05f, 7f);
                go.GetComponent<Renderer>().material.color = new Color(0.75f, 0.75f, 0.7f);
                labelHeight = 0.3f;
            }
            else
            {
                go = GameObject.CreatePrimitive(PrimitiveType.Cube);
                go.transform.position = new Vector3(x, 0.75f, z);
                go.transform.localScale = new Vector3(2.5f, 1.5f, 2.5f);
                go.GetComponent<Renderer>().material.color = kind == "bakery"
                    ? new Color(0.9f, 0.6f, 0.25f)
                    : new Color(0.3f, 0.3f, 0.35f);
                labelHeight = 1.6f;
            }
            go.name = name;

            // 名牌不挂在建筑下面（建筑被缩放过，子物体会跟着变形），直接放在世界坐标里
            var label = TextLabel.Make(null, name, new Vector3(x, labelHeight, z), 0.15f, Color.white);
            if (Camera.main != null) label.transform.rotation = Camera.main.transform.rotation;
        }
    }
}
