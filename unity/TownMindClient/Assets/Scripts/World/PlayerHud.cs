using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace TownMind.World
{
    /// <summary>
    /// 屏幕底部的聊天输入框，纯代码搭建——跟 TownWorld 搭镇子的风格一致，不用在 Unity
    /// 编辑器里手动拖 Canvas/InputField，打开工程按 Play 就能用。中文输入用系统字体，
    /// 跟 TextLabel 里对话气泡的做法一样（内置字体不含中文）。
    /// </summary>
    public static class PlayerHud
    {
        public static InputField Create()
        {
            if (Object.FindFirstObjectByType<EventSystem>() == null)
            {
                var es = new GameObject("EventSystem");
                es.AddComponent<EventSystem>();
                es.AddComponent<StandaloneInputModule>();
            }

            var canvasGo = new GameObject("PlayerHud");
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1280, 720);
            canvasGo.AddComponent<GraphicRaycaster>();

            var font = Font.CreateDynamicFontFromOSFont(new[] { "Microsoft YaHei", "SimHei", "Arial" }, 24);

            var bar = new GameObject("InputBar", typeof(RectTransform));
            bar.transform.SetParent(canvasGo.transform, false);
            var barRect = bar.GetComponent<RectTransform>();
            barRect.anchorMin = new Vector2(0, 0);
            barRect.anchorMax = new Vector2(1, 0);
            barRect.pivot = new Vector2(0.5f, 0);
            barRect.sizeDelta = new Vector2(0, 48);
            barRect.anchoredPosition = new Vector2(0, 12);

            var fieldGo = new GameObject("Input", typeof(RectTransform));
            fieldGo.transform.SetParent(bar.transform, false);
            var fieldRect = fieldGo.GetComponent<RectTransform>();
            fieldRect.anchorMin = new Vector2(0, 0);
            fieldRect.anchorMax = new Vector2(0.85f, 1);
            fieldRect.offsetMin = new Vector2(12, 0);
            fieldRect.offsetMax = new Vector2(-4, 0);
            var fieldImage = fieldGo.AddComponent<Image>();
            fieldImage.color = new Color(1f, 1f, 1f, 0.92f);
            var inputField = fieldGo.AddComponent<InputField>();

            var textGo = new GameObject("Text", typeof(RectTransform));
            textGo.transform.SetParent(fieldGo.transform, false);
            var textRect = textGo.GetComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = new Vector2(8, 4);
            textRect.offsetMax = new Vector2(-8, -4);
            var text = textGo.AddComponent<Text>();
            text.font = font;
            text.fontSize = 20;
            text.color = Color.black;
            text.alignment = TextAnchor.MiddleLeft;
            text.supportRichText = false;

            var placeholderGo = new GameObject("Placeholder", typeof(RectTransform));
            placeholderGo.transform.SetParent(fieldGo.transform, false);
            var placeholderRect = placeholderGo.GetComponent<RectTransform>();
            placeholderRect.anchorMin = Vector2.zero;
            placeholderRect.anchorMax = Vector2.one;
            placeholderRect.offsetMin = new Vector2(8, 4);
            placeholderRect.offsetMax = new Vector2(-8, -4);
            var placeholder = placeholderGo.AddComponent<Text>();
            placeholder.font = font;
            placeholder.fontSize = 20;
            placeholder.color = new Color(0, 0, 0, 0.4f);
            placeholder.text = "对附近的 NPC 说点什么，回车发送…";
            placeholder.alignment = TextAnchor.MiddleLeft;

            inputField.textComponent = text;
            inputField.placeholder = placeholder;

            var btnGo = new GameObject("SendButton", typeof(RectTransform));
            btnGo.transform.SetParent(bar.transform, false);
            var btnRect = btnGo.GetComponent<RectTransform>();
            btnRect.anchorMin = new Vector2(0.85f, 0);
            btnRect.anchorMax = new Vector2(1, 1);
            btnRect.offsetMin = new Vector2(4, 0);
            btnRect.offsetMax = new Vector2(-12, 0);
            var btnImage = btnGo.AddComponent<Image>();
            btnImage.color = new Color(0.29f, 0.49f, 0.29f, 1f);
            var button = btnGo.AddComponent<Button>();
            button.targetGraphic = btnImage;

            var btnTextGo = new GameObject("Text", typeof(RectTransform));
            btnTextGo.transform.SetParent(btnGo.transform, false);
            var btnTextRect = btnTextGo.GetComponent<RectTransform>();
            btnTextRect.anchorMin = Vector2.zero;
            btnTextRect.anchorMax = Vector2.one;
            btnTextRect.offsetMin = Vector2.zero;
            btnTextRect.offsetMax = Vector2.zero;
            var btnText = btnTextGo.AddComponent<Text>();
            btnText.font = font;
            btnText.fontSize = 20;
            btnText.color = Color.white;
            btnText.alignment = TextAnchor.MiddleCenter;
            btnText.text = "发送";

            button.onClick.AddListener(() =>
            {
                var pc = Object.FindFirstObjectByType<PlayerController>();
                if (pc != null) pc.SubmitText();
            });

            return inputField;
        }
    }
}
