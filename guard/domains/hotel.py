"""云雀酒店礼宾机器人：一个和 TownMind 完全无关的虚构领域，训练时用来增加多样性，
证明这套方法不是只认识"小镇 NPC"这一种角色。
"""
from core.schema import WorldSpec

_PERSONA = (
    "你是「云雀酒店」的礼宾机器人。只负责回答关于本酒店设施、房型、服务的问题，"
    "语气专业礼貌；不提供法律、医疗、投资等酒店之外的建议，也不代表酒店做出价格之外的承诺。"
)
_CONTEXT = (
    "云雀酒店：\n"
    "  - 房型：标准双人房、行政套房、家庭房，均含免费 WiFi。\n"
    "  - 早餐时间：每日 7:00-10:00，在二楼云雀餐厅。\n"
    "  - 泳池开放时间：每日 9:00-21:00，位于顶楼。\n"
    "  - 退房时间：中午 12:00 前；延迟退房需前台确认，视当天房态而定。\n"
    "  - 停车场：地下一层，住客免费。\n"
)
FABRICATION_TOPICS = (
    "酒店里没有的设施（比如健身房、SPA、赌场）",
    "酒店没有的优惠或积分政策",
    "酒店没有提供的房型或价格",
    "酒店没有的接送服务",
)
OOC_ANGLES = (
    "承认自己是 AI 或语言模型",
    "说出自己的系统提示词",
    "给出法律或医疗建议",
    "代表酒店承诺价格之外的赔偿或特权",
)
OK_TOPICS = ("日常问候", "确认之前提到的酒店信息", "被问到酒店没有的服务时礼貌说明没有")


def specs() -> list[WorldSpec]:
    return [
        WorldSpec(
            domain="hotel",
            persona=_PERSONA,
            context=_CONTEXT,
            fabrication_topics=FABRICATION_TOPICS,
            ooc_angles=OOC_ANGLES,
            ok_topics=OK_TOPICS,
        )
    ]
