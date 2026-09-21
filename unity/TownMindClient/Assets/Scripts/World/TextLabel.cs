using UnityEngine;

namespace TownMind.World
{
    /// <summary>创建能显示中文的 3D 文字（对话气泡、建筑名牌都用它）。</summary>
    public static class TextLabel
    {
        // 内置字体不含中文，改用系统字体（Windows 上是微软雅黑）。字体只创建一次，反复使用。
        private static Font _font;

        private static Font GetFont()
        {
            if (_font == null)
                _font = Font.CreateDynamicFontFromOSFont(new[] { "Microsoft YaHei", "SimHei", "Arial" }, 48);
            return _font;
        }

        public static TextMesh Make(Transform parent, string text, Vector3 localPos, float characterSize, Color color)
        {
            var go = new GameObject("Label");
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;

            var tm = go.AddComponent<TextMesh>();
            tm.text = text;
            tm.characterSize = characterSize;
            tm.fontSize = 48;
            tm.anchor = TextAnchor.LowerCenter;
            tm.alignment = TextAlignment.Center;
            tm.color = color;
            tm.font = GetFont();
            go.GetComponent<MeshRenderer>().material = GetFont().material;
            return tm;
        }
    }
}
