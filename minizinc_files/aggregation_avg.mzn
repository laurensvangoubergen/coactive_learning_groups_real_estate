%configuration for a real estate object with features:
% - type: house, flat, ...
% - area: city rural, ..
% - rooms: the number of every type of room e.g. 4 bedrooms, 2 bathrooms, ... (living room and kitchen are set)
% - layout: defined by layout_features and layout options
% - price: the maximum price

int: nb_users;
int: nb_types;
int: nb_areas;
int: nb_room_types;
int: nb_of_rooms;
int: nb_layout_features;
int: nb_layout_options;
set of int: USERS = 1..nb_users;
set of int: TYPES = 1..nb_types;
set of int: AREAS = 1..nb_areas;
set of int: ROOM_TYPES = 1..nb_room_types;
set of int: ROOMS = 1..nb_of_rooms;
set of int: LAYOUT_FEATURES = 1..nb_layout_features;
set of int: LAYOUT_OPTIONS = 1..nb_layout_options;
int: total;
int: price;


%array representing the type weights (given)
array[USERS, TYPES] of float: type_weights;

%array representing the area weights (given)
array[USERS, AREAS] of float: area_weights;

%array representing the rooms weights (given)
array[USERS, ROOM_TYPES, ROOMS] of float: room_weights;

%array representing the layout_weights (given)
%e.g.: w = [9, 10]
array[USERS, LAYOUT_FEATURES, LAYOUT_OPTIONS] of float: layout_weights;

%phi types
array[TYPES] of var 0..1: type_terms;
constraint sum (t in TYPES) (type_terms[t]) = 1;

%phi area
array[AREAS] of var 0..1: area_terms;
constraint sum (a in AREAS) (area_terms[a]) = 1;

%phi rooms
array[ROOM_TYPES, ROOMS] of var 0..1: room_terms;
constraint forall(t in ROOM_TYPES)  (1 = sum(r in ROOMS) (room_terms[t,r]));

%2d representing the actual features and their options, in a one-hot fashion
array[LAYOUT_FEATURES, LAYOUT_OPTIONS] of var 0..1: layout_terms;

%make sure the options are all one hot vectors
constraint forall(f in LAYOUT_FEATURES) (1 = sum (o in LAYOUT_OPTIONS) (layout_terms[f,o]));

%%%%%FEATURE REPRESENTATIONS%%%%%

var 0..nb_types: type_feature_rep;

constraint type_feature_rep = sum( [ if 1 = type_terms[i] then i else 0 endif | i in TYPES] );

var 0..nb_areas: area_feature_rep;
constraint area_feature_rep = sum( [ if 1 = area_terms[i] then i else 0 endif | i in AREAS] );

%array to store the feature representation
array[ROOM_TYPES] of var 0..nb_room_types: rooms_feature_rep;

constraint forall(f in ROOM_TYPES) (rooms_feature_rep[f] = sum ( [ if 1 = room_terms[f,i]
							then i
						    else 0 endif
						    | i in ROOMS] ));

%array to store the feature representation
array[LAYOUT_FEATURES] of var 0..nb_layout_options: layout_feature_rep;

constraint forall(f in LAYOUT_FEATURES) (layout_feature_rep[f] = sum ( [ if 1 = layout_terms[f,i]
							then i-1
						    else 0 endif
						    | i in LAYOUT_OPTIONS] ));

%make sure the rooms are equal to the number of rooms in room types plus a living room and a kitchen

array[LAYOUT_FEATURES] of var 0..1 : rooms;
constraint forall(f in LAYOUT_FEATURES) (rooms[f] = sum([ if layout_feature_rep[f] > 0 then 1 else 0 endif ]));
var int: nb_r;
constraint nb_r = sum(rooms);

constraint sum(rooms) = sum(rooms_feature_rep) + 1;

%%%%%%%%%%%%CONSTRAINTS%%%%%%%%%%%%%%%%%%%%%%%
%living room must actually be rooms (no zeros)
constraint layout_feature_rep[1] > 0;


%constraint to limit the area to total
constraint total >= sum(f in LAYOUT_FEATURES) (layout_feature_rep[f]);

%the price of the estate
var int: estate_price;
constraint estate_price = (70000 * type_feature_rep) + (30000 * area_feature_rep) + (sum(t in ROOM_TYPES) (rooms_feature_rep[t] * 20000));
constraint price >= estate_price;

%assuming the first element of the layout is the living room, this is the largest room
constraint forall (f in LAYOUT_FEATURES) (layout_feature_rep[1] >= layout_feature_rep[f]);


%UTILITIES%
%the utility functions: the weighted sum for the terms

function var float: utility_type(array[USERS, TYPES] of float: w, int: u) =
        (sum (t in TYPES) (w[u,t] * type_terms[t]));

function var float: utility_area(array[USERS, AREAS] of float: w, int:u) =
        (sum (a in AREAS) (w[u,a] * area_terms[a]));

function var float: utility_rooms(array[USERS, ROOM_TYPES, ROOMS] of float: w, int:u) =
        (sum (t in ROOM_TYPES) (sum (r in ROOMS) (w[u,t,r] * room_terms[t,r])));

function var float: utility_layout(array[USERS, LAYOUT_FEATURES, LAYOUT_OPTIONS] of float: w, int: u) =
        (sum (f in LAYOUT_FEATURES) (sum (o in LAYOUT_OPTIONS) (w[u,f,o] * layout_terms[f,o])));

%constraint forall(u in USERS) (utility_user(weights,u) = utilities[u]);
function var float: utility_combined(int: u) =
        (utility_type(type_weights, u) + utility_area(area_weights, u) + utility_rooms(room_weights, u)
        + utility_layout(layout_weights, u));



%AGGREGATION%
%the aggregation functions

function var float: aggregate() = (sum(u in USERS) (utility_combined(u))) / nb_users;



var float: agg;
constraint agg = aggregate();


%solve maximize utility(weights);
solve maximize aggregate();
%solve satisfy;

