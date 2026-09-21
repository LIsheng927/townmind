using TownMind.Net;
using UnityEngine;

namespace TownMind.World
{
    /// <summary>
    /// 用代码搭建最小小镇：地面 + 3 个方块 NPC + 俯视相机。
    /// 挂在和 TownClient 相同的物体上。
    /// </summary>
    [RequireComponent(typeof(TownClient))]
    public class TownWorld : MonoBehaviour
    {
        private void Start()
        {
            var client = GetComponent<TownClient>();

            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.localScale = new Vector3(2f, 1f, 2f); // Plane 默认 10x10，这里变成 20x20
            ground.GetComponent<Renderer>().material.color = new Color(0.35f, 0.5f, 0.3f);

            Spawn(client, "alice", new Vector3(-3, 0.5f, 0), Color.red);
            Spawn(client, "bob", new Vector3(0, 0.5f, 3), Color.blue);
            Spawn(client, "carol", new Vector3(3, 0.5f, -2), Color.yellow);

            var cam = Camera.main;
            if (cam != null)
            {
                cam.transform.position = new Vector3(0, 16, -12);
                cam.transform.rotation = Quaternion.Euler(55, 0, 0);
            }
        }

        private static void Spawn(TownClient client, string id, Vector3 pos, Color color)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = id;
            go.transform.position = pos;
            go.GetComponent<Renderer>().material.color = color;
            go.AddComponent<NpcController>().Init(id, client);
        }
    }
}
