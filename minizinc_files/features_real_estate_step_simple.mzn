%configuration for a real estate object with features:
% - type: house, flat, ...
% - area: city rural, ..
% - rooms: the number of every type of room e.g. 4 bedrooms, 2 bathrooms, ... (living room and kitchen are set)
% - layout: defined by layout_features and layout options
% - price: the maximum price

int: nb_types;
int: nb_areas;
int: nb_room_types;
int: nb_of_rooms;
int: nb_layout_features;
int: nb_layout_options;
set of int: TYPES = 1..nb_types;
set of int: AREAS = 1..nb_areas;
set of int: ROOM_TYPES = 1..nb_room_types;
set of int: ROOMS = 1..nb_of_rooms;
set of int: LAYOUT_FEATURES = 1..nb_layout_features;
set of int: LAYOUT_OPTIONS = 1..nb_layout_options;
int: total;
int: price;
float: desired_utility;

%%%%%%%ALL WEIGHTS ARE TARGET WEIGHTS%%%%%%%%%
%array representing the type weights (given)
array[TYPES] of float: type_weights;

%array representing the area weights (given)
array[AREAS] of float: area_weights;

%array representing the rooms weights (given)
array[ROOM_TYPES, ROOMS] of float: room_weights;

%array representing the layout_weights (given)
%e.g.: w = [9, 10]
array[LAYOUT_FEATURES, LAYOUT_OPTIONS] of float: layout_weights;


%%%%%%%%%%%%PHI BARS%%%%%%%%%%%%%%%%
%phi types
array[TYPES] of var 0..1: type_terms_bar;
constraint sum (t in TYPES) (type_terms_bar[t]) = 1;

%phi area
array[AREAS] of var 0..1: area_terms_bar;
constraint sum (a in AREAS) (area_terms_bar[a]) = 1;

%phi rooms
array[ROOM_TYPES, ROOMS] of var 0..1: room_terms_bar;
constraint forall(t in ROOM_TYPES)  (1 = sum(r in ROOMS) (room_terms_bar[t,r]));

%phi layout
array[LAYOUT_FEATURES, LAYOUT_OPTIONS] of var 0..1: layout_terms_bar;
constraint forall(f in LAYOUT_FEATURES) (1 = sum (o in LAYOUT_OPTIONS) (layout_terms_bar[f,o]));


%%%%%FEATURE REPRESENTATIONS%%%%%

int: type_feature_rep;

int: area_feature_rep;

array[ROOM_TYPES] of int: rooms_feature_rep;

array[LAYOUT_FEATURES] of int: layout_feature_rep;

%%%%%FEATURE REPRESENTATIONS BAR%%%%%

var 0..nb_types: type_feature_rep_bar;

constraint type_feature_rep_bar = sum( [ if 1 = type_terms_bar[i] then i else 0 endif | i in TYPES] );

var 0..nb_areas: area_feature_rep_bar;
constraint area_feature_rep_bar = sum( [ if 1 = area_terms_bar[i] then i else 0 endif | i in AREAS] );

%array to store the feature representation
array[ROOM_TYPES] of var 0..nb_room_types: rooms_feature_rep_bar;
constraint forall(f in ROOM_TYPES) (rooms_feature_rep_bar[f] = sum ( [ if 1 = room_terms_bar[f,i]
							then i
						    else 0 endif
						    | i in ROOMS] ));

%array to store the feature representation
array[LAYOUT_FEATURES] of var 0..nb_layout_options: layout_feature_rep_bar;
constraint forall(f in LAYOUT_FEATURES) (layout_feature_rep_bar[f] = sum ( [ if 1 = layout_terms_bar[f,i]
							then i-1
						    else 0 endif
						    | i in LAYOUT_OPTIONS] ));

%make sure the rooms are equal to the number of rooms in room types plus a living room and a kitchen

array[LAYOUT_FEATURES] of var 0..1 : rooms;
constraint forall(f in LAYOUT_FEATURES) (rooms[f] = sum([ if layout_feature_rep_bar[f] > 0 then 1 else 0 endif ]));
%var int: nb_r;
%constraint nb_r = sum(rooms);

constraint sum(rooms) == (sum(rooms_feature_rep_bar) + 1);

%%%%%%%%%%%%CONSTRAINTS%%%%%%%%%%%%%%%%%%%%%%%
%living room must actually be rooms (no zeros)
constraint layout_feature_rep_bar[1] > 0;


%constraint to limit the area to total
constraint total >= sum(f in LAYOUT_FEATURES) (layout_feature_rep_bar[f]);

%the price of the estate
var int: estate_price;
constraint estate_price = (70000 * type_feature_rep_bar) + (30000 * area_feature_rep_bar) + (sum(t in ROOM_TYPES) (rooms_feature_rep_bar[t] * 20000));
constraint price >= estate_price;

%assuming the first element of the layout is the living room, this is the largest room
constraint forall (f in LAYOUT_FEATURES) (layout_feature_rep[1] >= layout_feature_rep[f]);

%UTILITIES%

function var float: utility_type(array[TYPES] of float: w, array[TYPES] of var int: phi) =
        (sum (t in TYPES) (w[t] * phi[t]));

function var float: utility_area(array[AREAS] of float: w, array[AREAS] of var int: phi) =
        (sum (a in AREAS) (w[a] * phi[a]));

function var float: utility_rooms(array[ROOM_TYPES, ROOMS] of float: w, array[ROOM_TYPES, ROOMS] of var int: phi) =
        (sum (t in ROOM_TYPES) (sum (r in ROOMS) (w[t,r] * phi[t,r])));

function var float: utility_layout(array[LAYOUT_FEATURES, LAYOUT_OPTIONS] of float: w, array[LAYOUT_FEATURES, LAYOUT_OPTIONS] of var int: phi) =
        (sum (f in LAYOUT_FEATURES) (sum (o in LAYOUT_OPTIONS) (w[f,o] * phi[f,o])));

var float: utility_bar = (utility_rooms(room_weights, room_terms_bar) + utility_layout(layout_weights, layout_terms_bar)
                            + utility_type(type_weights, type_terms_bar) + utility_area(area_weights, area_terms_bar));



constraint utility_bar >= desired_utility;

%solve satisfy;
%var int: dist_type = sqrt((type_feature_rep_bar - type_feature_rep)^2);
%var int: dist_area = sqrt((area_feature_rep_bar - area_feature_rep)^2);
%var int: dist_rooms = sqrt( sum(t in ROOM_TYPES) ((rooms_feature_rep_bar[t] - rooms_feature_rep[t])^2));
%var int: dist_layout = sqrt(sum(f in LAYOUT_FEATURES) ((layout_feature_rep_bar[f] - layout_feature_rep[f])));

var int: dist_type = abs(type_feature_rep_bar - type_feature_rep);
var int: dist_area = abs(area_feature_rep_bar - area_feature_rep);
var int: dist_rooms = sum(t in ROOM_TYPES) (abs(rooms_feature_rep_bar[t] - rooms_feature_rep[t]));
var int: dist_layout = sum(f in LAYOUT_FEATURES) (abs(layout_feature_rep_bar[f] - layout_feature_rep[f]));

solve minimize (dist_type + dist_area + dist_rooms + dist_layout);
%solve minimize sqrt( sum(f in FEATURES) (abs(feature_rep_x_bar[f] - feature_rep_x[f])^2));