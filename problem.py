class Problem:
    def __init__(self, types, areas, features, options, upper_price, roomtypes, max_rooms, users, iterations):
        self.types = types
        self.areas = areas
        self.upper_price = upper_price
        self.roomtypes = roomtypes
        self.max_rooms = max_rooms
        self.users = users
        self.iterations = iterations
        self.features = features
        self.options = options

    def __getattr__(self, item):
        return item
