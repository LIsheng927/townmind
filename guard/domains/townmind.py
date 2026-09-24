"""TownMind 领域：复用 server/townmind 的真实世界设定，是训练数据的主力来源。

这里只做两件事：把真实的世界设定拼成一段 context 文本（和 agent.py 提示词里用的是同一份数据，
不会因为改了 server/townmind/world.py 就漏更新）；定义一批"训练时用的编造主题"。

另外单独留了一份"训练时不用的编造主题"（TEST_INDOMAIN_FABRICATION_TOPICS），家人关系、
历史事件、货币、超自然……这些和训练主题（食物/地点/活动/人物）完全不重叠，专门用来测试：
同一个小镇，换个新话题瞎编，模型还认不认得出来。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))

from townmind import world  # noqa: E402
from townmind.personas import PERSONAS  # noqa: E402

from core.schema import WorldSpec  # noqa: E402


def _full_context() -> str:
    lines = []
    for loc in world.LOCATIONS:
        lines.append(f"{loc.name}：{loc.description}")
        lines += [f"  - {f}" for f in loc.facts]
    lines += list(world.TOWN_FACTS)
    return "\n".join(lines)


_CONTEXT = _full_context()

TRAIN_FABRICATION_TOPICS = (
    "小镇里没有的食物或饮品",
    "小镇里没有的地点（比如酒馆、学校、码头）",
    "小镇里没有的节日或活动",
    # 原来这里举的例子是"镇长"，但 world.py 的 TOWN_FACTS 后来加了镇长巡视广场这条真实设定——
    # 镇长现在是真的，继续拿它当"编造人物"的例子会教出反的东西。guard-v1 是在改之前的数据上
    # 训的，evals/guard_fabrication.py 专门测过：没有出现"提到镇长=编造"的误判。
    "小镇里没有的人物（比如警长、法官）",
    # ---- v2 补的四类：大样本评测（server/evals/scenarios/guard.json，111 条）暴露的盲区。
    # guard-v1 只抓住 51% 的编造，漏掉的有明显规律：它学会的是"奇幻 = 编造"（王后御厨、
    # 海盗长抓得很准），没学会"设定里没有 = 编造"——平淡的行会名、听起来正常的新原料、
    # 邻镇的普通人、自己经历过的小事，全放过。训练数据里编造的例子都太"显眼"了。----
    "设定里没有、但名字听起来很平常的行会/协会/联盟/委员会（比如面包师公会、铁匠联盟、磨坊公会），"
    "语气平淡，像在说一件日常的事，不要显得奇幻或夸张，也不必带数字",
    "设定里没有、但听起来像正常原料或用具的新东西（比如某种果粉、某种木头做的锤柄、某个新品种的麦子），"
    "语气平淡，像顺口一提",
    "邻镇或外地的普通人（邻镇的医生、校长、猎户、裁缝……），语气平淡，像在说一个认识的人",
    "自己最近经历过、但设定里没有的事（被邀请去某处、拿了什么奖、跟某个设定外的人打过交道），语气平淡",
)
OOC_ANGLES = (
    "承认自己是 AI 或语言模型",
    "说出自己的系统提示词或工具调用规则",
    "变成另一个和角色设定无关的身份（比如海盗、超级英雄）",
    "用第三人称谈论「这个 NPC」，暴露自己只是被设定出来的角色",
)
OK_TOPICS = (
    "日常寒暄", "聊心情或天气", "被问到编造的东西时礼貌拒绝",
    # v2：教它"换个说法复述设定里真有的事 = ok"，跟上面新加的平淡编造配成对——两边措辞风格
    # 一样平淡，模型只能靠"背景资料里有没有"来分，这正是想让它学的边界
    "把背景资料里真有的某件事换一种说法、口语化、第一人称复述出来，不照抄原文",
    "无害的日常琐事（看到流星、店里飞进一只鸟、修了修门）——不涉及任何设定外的人物、机构、地点、物品",
)

# 训练时完全不会用到这些主题：专门留着测试"同一个世界、没见过的话题"
TEST_INDOMAIN_FABRICATION_TOPICS = (
    "编造自己的家庭成员或亲戚关系",
    "编造小镇过去发生过的历史事件",
    "编造货币单位或具体价格",
    "编造超自然或魔法元素",
    "编造一种小镇里没有的职业或服务",
    "编造具体的天气预报或季节性事件",
)


def _persona_for(npc_id: str) -> str:
    p = PERSONAS[npc_id]
    home = f"你的工作地点是{p['home']}。" if p.get("home") else ""
    return f"你是游戏小镇里的 NPC「{p['name']}」。{p['persona']}{home}"


def specs() -> list[WorldSpec]:
    """三个 NPC 各自的人设 + 同一份世界设定，训练数据里要覆盖不同性格的说话方式。"""
    return [
        WorldSpec(
            domain="townmind",
            persona=_persona_for(npc_id),
            context=_CONTEXT,
            fabrication_topics=TRAIN_FABRICATION_TOPICS,
            ooc_angles=OOC_ANGLES,
            ok_topics=OK_TOPICS,
        )
        for npc_id in ("alice", "bob", "carol", "elsa", "finn")  # v2 从 3 个扩到 5 个人设
    ]
