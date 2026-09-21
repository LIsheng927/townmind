import math

from townmind import world


def test_all_locations_inside_world_bounds():
    for loc in world.LOCATIONS:
        assert abs(loc.x) <= 8 and abs(loc.z) <= 8


def test_stand_point_is_in_bounds_and_counts_as_being_there():
    for loc in world.LOCATIONS:
        for _ in range(50):
            x, z = loc.stand_point()
            assert abs(x) <= 8 and abs(z) <= 8
            assert world.location_at((x, z)) == loc


def test_open_ground_is_not_a_location():
    assert world.location_at((0, 3)) is None


def test_get_location():
    assert world.get_location("面包店").id == "bakery"
    assert world.get_location("月球") is None


def test_describe_surroundings_at_a_location():
    text = "\n".join(world.describe_surroundings((-5, 2)))
    assert "你现在在：面包店" in text and "面粉涨价" in text and "镇上的事" in text


def test_describe_surroundings_in_open_ground_names_nearest():
    text = "\n".join(world.describe_surroundings((0, 3)))
    assert "空地" in text and "最近" in text
    assert "面粉涨价" not in text


def test_locations_payload_shape():
    payload = world.locations_payload()
    assert len(payload) == len(world.LOCATIONS)
    assert set(payload[0]) == {"id", "name", "x", "z", "kind"}
