#map = affine_map<()[s0, s1, s2] -> ((s1 + s2 * 32 + s0 floordiv 32) mod 32)>
#map1 = affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 32) * 4096)>
#map2 = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>
#map3 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>
#map4 = affine_map<(d0) -> (d0 floordiv 2)>
#map5 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>
#map6 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>
#map7 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>
#map8 = affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>
#map9 = affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128)>
#map10 = affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 1)>
#map11 = affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2)>
#map12 = affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 3)>
#map13 = affine_map<(d0)[s0] -> (d0 * 2048 - ((d0 + 1) floordiv 2) * 3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>
#map14 = affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>
#map15 = affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>
#map16 = affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>
#map17 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>
#map18 = affine_map<(d0) -> (d0 floordiv 8)>
#map19 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>
#map20 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>
#map21 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>
#map22 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>
#map23 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>
#map24 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>
#map25 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>
#map26 = affine_map<()[s0] -> (s0 + 1)>
#map27 = affine_map<(d0)[s0, s1] -> ((d0 floordiv 4096 + s0) mod 32)>
#map28 = affine_map<(d0) -> ((d0 floordiv 32) mod 128)>
#map29 = affine_map<(d0) -> (d0 * 128 - (d0 floordiv 32) * 4096)>
#map30 = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>
#map31 = affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 1024) floordiv 4096) mod 32)>
#map32 = affine_map<(d0)[s0, s1] -> (d0 floordiv 32 - ((d0 floordiv 32 + s0 * 128 + s1 * 4096 + 32) floordiv 128) * 128 + s0 * 128 + s1 * 4096 + 32)>
#map33 = affine_map<(d0)[s0] -> (d0 * 32 - (d0 floordiv 8) * 512 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 1024 - ((d0 mod 8) floordiv 4) * 256 + s0 * 256 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 mod 8 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 16 - ((d0 mod 8) floordiv 4) * 4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>
#map34 = affine_map<(d0)[s0] -> (d0 * 32 - (d0 floordiv 8) * 512 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 1024 - ((d0 mod 8) floordiv 4) * 256 + s0 * 256 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 mod 8 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 16 - ((d0 mod 8) floordiv 4) * 4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>
#map35 = affine_map<(d0)[s0] -> (d0 * 32 - (d0 floordiv 8) * 512 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 1024 - ((d0 mod 8) floordiv 4) * 256 + s0 * 256 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 mod 8 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 16 - ((d0 mod 8) floordiv 4) * 4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>
#map36 = affine_map<(d0)[s0] -> (d0 * 32 - (d0 floordiv 8) * 512 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 1024 - ((d0 mod 8) floordiv 4) * 256 + s0 * 256 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 mod 8 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 16 - ((d0 mod 8) floordiv 4) * 4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>
#map37 = affine_map<(d0)[s0] -> (d0 * 32 - (d0 floordiv 8) * 512 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 1024 - ((d0 mod 8) floordiv 4) * 256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 mod 8 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 16 - ((d0 mod 8) floordiv 4) * 4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + 2048)>
#map38 = affine_map<(d0)[s0] -> (d0 * 32 - (d0 floordiv 8) * 256 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 512 - ((d0 mod 8) floordiv 4) * 128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((s0 mod 16) floordiv 4) * 8 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>
#map39 = affine_map<(d0)[s0] -> (d0 * 32 - (d0 floordiv 8) * 256 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 512 - ((d0 mod 8) floordiv 4) * 128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((s0 mod 16) floordiv 4) * 8 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>
#map40 = affine_map<(d0)[s0] -> (d0 * 32 - (d0 floordiv 8) * 256 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 512 - ((d0 mod 8) floordiv 4) * 128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + s0 mod 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((s0 mod 16) floordiv 4) * 8 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>
#map41 = affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 2048) floordiv 4096) mod 32)>
#map42 = affine_map<(d0)[s0, s1] -> (d0 floordiv 32 - ((d0 floordiv 32 + s0 * 128 + s1 * 4096 + 64) floordiv 128) * 128 + s0 * 128 + s1 * 4096 + 64)>
#map43 = affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 3072) floordiv 4096) mod 32)>
#map44 = affine_map<(d0)[s0, s1] -> (d0 floordiv 32 - ((d0 floordiv 32 + s0 * 128 + s1 * 4096 + 96) floordiv 128) * 128 + s0 * 128 + s1 * 4096 + 96)>
#map45 = affine_map<(d0)[s0, s1] -> ((d0 floordiv 32 + s0) mod 32)>
#map46 = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>
#map47 = affine_map<(d0)[s0, s1] -> ((d0 floordiv 8) * 64 + s0 + s1 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>
#map48 = affine_map<(d0, d1)[s0] -> (d0 * 4 + d1 * 128 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16) floordiv 64) * 64 - ((d0 mod 8) floordiv 4) * 16 + ((s0 mod 64) floordiv 16) * 16)>
#map49 = affine_map<(d0, d1)[s0] -> (d1 * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 4) floordiv 64) * 64 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 4) floordiv 16) * 4 - (((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 4) floordiv 16) floordiv 4) * 16 + ((s0 mod 64) floordiv 16) * 16 + 1)>
#map50 = affine_map<(d0, d1)[s0] -> (d1 * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 8) floordiv 64) * 64 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 8) floordiv 16) * 4 - (((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 8) floordiv 16) floordiv 4) * 16 + ((s0 mod 64) floordiv 16) * 16 + 2)>
#map51 = affine_map<(d0, d1)[s0] -> (d1 * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 12) floordiv 64) * 64 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 12) floordiv 16) * 4 - (((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 12) floordiv 16) floordiv 4) * 16 + ((s0 mod 64) floordiv 16) * 16 + 3)>
#map52 = affine_map<(d0) -> (d0 * 4)>
#map53 = affine_map<(d0) -> (d0 * 4 + 1)>
#map54 = affine_map<(d0) -> (d0 * 4 + 2)>
#map55 = affine_map<(d0) -> (d0 * 4 + 3)>
#map56 = affine_map<(d0)[s0] -> ((d0 floordiv 32 + s0) mod 32)>
#map57 = affine_map<(d0)[s0] -> (d0 * 128 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>
#map58 = affine_map<(d0) -> (d0 mod 2)>
#map59 = affine_map<()[s0] -> (s0 floordiv 64)>
module {
  func.func @Attn(%arg0: tensor<1x4096x32x128xf16>, %arg1: tensor<1x4096x32x128xf16>, %arg2: tensor<1x4096x32x128xf16>) -> tensor<1x4096x32x128xf16> {
    %cst = arith.constant dense<1.131250e+01> : tensor<1xf32>
    %0 = deepgengraph.trilu diagonal = 1, is_upper = true, shape = [4096, 4096], val = 0xFF800000 : f32
    %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %2 = deepgengraph.permute %arg2, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %3 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x128x4096xf16>
    %4 = deepgengraph.precise_dot_op %1, %3, acc = f32 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf32>
    %5 = deepgengraph.div %4, %cst : (tensor<1x32x4096x4096xf32>, tensor<1xf32>) -> tensor<1x32x4096x4096xf32>
    %6 = deepgengraph.add %5, %0 : (tensor<1x32x4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %7 = deepgengraph.reduce(%6), dim = -1, op = MAX, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
    %8 = deepgengraph.sub %6, %7 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
    %9 = deepgengraph.exp %8 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %10 = deepgengraph.reduce(%9), dim = -1, op = ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
    %11 = deepgengraph.div %9, %10 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
    %12 = deepgengraph.convert %11, type = f16 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16>
    %13 = deepgengraph.dot %12, %2 : (tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16>
    %14 = deepgengraph.permute %13, dims = [0, 2, 1, 3] : (tensor<1x32x4096x128xf16>) -> tensor<1x4096x32x128xf16>
    return %14 : tensor<1x4096x32x128xf16>
  }
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 4, 0, 1, 1>, thread_num = 256 : i32, warp_layout = array<i64: 16, 4>} {
    %c8192 = arith.constant 8192 : index
    %cst = arith.constant 0.127531052 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %cst_3 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c31 = arith.constant 31 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() alignment = 16 {frisk.shm_pool, tiled = true} : memref<16384xi8, 3>
    %block_id_x = gpu.block_id x {range = 32 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %0 = arith.muli %block_id_y, %c32 : index
    %1 = arith.addi %block_id_x, %0 : index
    %2 = arith.divui %1, %c128 : index
    %3 = arith.muli %2, %c4 : index
    %4 = arith.remui %1, %c128 : index
    %5 = arith.remui %4, %c4 : index
    %6 = arith.addi %3, %5 : index
    %7 = arith.subi %c31, %6 : index
    %8 = arith.divui %4, %c4 : index
    %block_id_z = gpu.block_id z {range = 1 : index}
    %9 = affine.apply #map()[%7, %8, %block_id_z]
    %10 = affine.apply #map1()[%7]
    %subview = memref.subview %arg0[0, %9, %10, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %view = memref.view %alloc[%c0][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %11 = frisk.async_copy_gm_to_shm %subview to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = #map2} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %12 = arith.truncf %cst : f32 to f16
    %13 = arith.muli %7, %c128 : index
    %14:4 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_1, %arg6 = %cst_1, %arg7 = %cst_1, %arg8 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %108 = vector.insert %cst_7, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %109 = vector.insert %cst_0, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %110 = vector.insert %cst_0, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      %111 = vector.insert %cst_0, %arg8 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %108, %109, %110, %111 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %15 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %108 = affine.apply #map3(%arg4)
      %109 = affine.apply #map4(%arg4)
      %110 = vector.insert %cst_7, %arg5 [%109, %108] : f32 into vector<2x8xf32>
      %111 = affine.apply #map5(%arg4)
      %112 = vector.insert %cst_7, %110 [%109, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply #map6(%arg4)
      %114 = vector.insert %cst_7, %112 [%109, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply #map7(%arg4)
      %116 = vector.insert %cst_7, %114 [%109, %115] : f32 into vector<2x8xf32>
      affine.yield %116 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %16 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_1) -> (vector<2x1xf32>) {
      %108 = vector.insert %cst_7, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %108 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
    frisk.wait_buffer_ready %11 : !frisk.async_task
    %view_8 = memref.view %alloc[%c8192][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %17 = affine.apply #map8()[%7, %8]
    %subview_9 = memref.subview %arg0[0, %17, %10, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %18 = frisk.async_copy_gm_to_shm %subview_9 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = #map2} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %thread_id_x = gpu.thread_id x
    %19 = affine.apply #map9()[%thread_id_x]
    %20 = memref.load %view[%19] : memref<4096xf16, 3>
    %21 = affine.apply #map10()[%thread_id_x]
    %22 = memref.load %view[%21] : memref<4096xf16, 3>
    %23 = affine.apply #map11()[%thread_id_x]
    %24 = memref.load %view[%23] : memref<4096xf16, 3>
    %25 = affine.apply #map12()[%thread_id_x]
    %26 = memref.load %view[%25] : memref<4096xf16, 3>
    %27 = vector.from_elements %20, %22, %24, %26 : vector<1x4xf16>
    %28 = vector.broadcast %12 : f16 to vector<1x4xf16>
    %29:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %27) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %108 = affine.apply #map13(%arg4)[%thread_id_x]
      %109 = memref.load %view[%108] : memref<4096xf16, 3>
      %110 = affine.apply #map14(%arg4)[%thread_id_x]
      %111 = memref.load %view[%110] : memref<4096xf16, 3>
      %112 = affine.apply #map15(%arg4)[%thread_id_x]
      %113 = memref.load %view[%112] : memref<4096xf16, 3>
      %114 = affine.apply #map16(%arg4)[%thread_id_x]
      %115 = memref.load %view[%114] : memref<4096xf16, 3>
      %116 = vector.from_elements %109, %111, %113, %115 : vector<1x4xf16>
      %117 = affine.apply #map4(%arg4)
      %118 = affine.apply #map3(%arg4)
      %119 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %arg5 [%117, %118] : f16 into vector<2x8xf16>
      %121 = affine.apply #map5(%arg4)
      %122 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %120 [%117, %121] : f16 into vector<2x8xf16>
      %124 = affine.apply #map6(%arg4)
      %125 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %123 [%117, %124] : f16 into vector<2x8xf16>
      %127 = affine.apply #map7(%arg4)
      %128 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %126 [%117, %127] : f16 into vector<2x8xf16>
      %130 = vector.insert %12, %arg6 [%117, %118] : f16 into vector<2x8xf16>
      %131 = vector.insert %12, %130 [%117, %121] : f16 into vector<2x8xf16>
      %132 = vector.insert %12, %131 [%117, %124] : f16 into vector<2x8xf16>
      %133 = vector.insert %12, %132 [%117, %127] : f16 into vector<2x8xf16>
      %134 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%117, %118] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%117, %121] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%117, %124] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%117, %127] : f16 into vector<2x8xf16>
      affine.yield %129, %133, %142, %116 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %30 = arith.mulf %29#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %31 = vector.extract %30[0, 0] : f16 from vector<1x4xf16>
    %32 = vector.insert %31, %29#2 [1, 4] : f16 into vector<2x8xf16>
    %33 = vector.extract %30[0, 1] : f16 from vector<1x4xf16>
    %34 = vector.insert %33, %32 [1, 5] : f16 into vector<2x8xf16>
    %35 = vector.extract %30[0, 2] : f16 from vector<1x4xf16>
    %36 = vector.insert %35, %34 [1, 6] : f16 into vector<2x8xf16>
    %37 = vector.extract %30[0, 3] : f16 from vector<1x4xf16>
    %38 = vector.insert %37, %36 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %18 : !frisk.async_task
    %subview_10 = memref.subview %arg0[0, %17, %10, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %39 = frisk.async_copy_gm_to_shm %subview_10 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = #map2} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %40 = memref.load %view_8[%19] : memref<4096xf16, 3>
    %41 = memref.load %view_8[%21] : memref<4096xf16, 3>
    %42 = memref.load %view_8[%23] : memref<4096xf16, 3>
    %43 = memref.load %view_8[%25] : memref<4096xf16, 3>
    %44 = vector.from_elements %40, %41, %42, %43 : vector<1x4xf16>
    %45:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %44) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %108 = affine.apply #map13(%arg4)[%thread_id_x]
      %109 = memref.load %view_8[%108] : memref<4096xf16, 3>
      %110 = affine.apply #map14(%arg4)[%thread_id_x]
      %111 = memref.load %view_8[%110] : memref<4096xf16, 3>
      %112 = affine.apply #map15(%arg4)[%thread_id_x]
      %113 = memref.load %view_8[%112] : memref<4096xf16, 3>
      %114 = affine.apply #map16(%arg4)[%thread_id_x]
      %115 = memref.load %view_8[%114] : memref<4096xf16, 3>
      %116 = vector.from_elements %109, %111, %113, %115 : vector<1x4xf16>
      %117 = affine.apply #map4(%arg4)
      %118 = affine.apply #map3(%arg4)
      %119 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %arg5 [%117, %118] : f16 into vector<2x8xf16>
      %121 = affine.apply #map5(%arg4)
      %122 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %120 [%117, %121] : f16 into vector<2x8xf16>
      %124 = affine.apply #map6(%arg4)
      %125 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %123 [%117, %124] : f16 into vector<2x8xf16>
      %127 = affine.apply #map7(%arg4)
      %128 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %126 [%117, %127] : f16 into vector<2x8xf16>
      %130 = vector.insert %12, %arg6 [%117, %118] : f16 into vector<2x8xf16>
      %131 = vector.insert %12, %130 [%117, %121] : f16 into vector<2x8xf16>
      %132 = vector.insert %12, %131 [%117, %124] : f16 into vector<2x8xf16>
      %133 = vector.insert %12, %132 [%117, %127] : f16 into vector<2x8xf16>
      %134 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%117, %118] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%117, %121] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%117, %124] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%117, %127] : f16 into vector<2x8xf16>
      affine.yield %129, %133, %142, %116 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %46 = arith.mulf %45#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %47 = vector.extract %46[0, 0] : f16 from vector<1x4xf16>
    %48 = vector.insert %47, %45#2 [1, 4] : f16 into vector<2x8xf16>
    %49 = vector.extract %46[0, 1] : f16 from vector<1x4xf16>
    %50 = vector.insert %49, %48 [1, 5] : f16 into vector<2x8xf16>
    %51 = vector.extract %46[0, 2] : f16 from vector<1x4xf16>
    %52 = vector.insert %51, %50 [1, 6] : f16 into vector<2x8xf16>
    %53 = vector.extract %46[0, 3] : f16 from vector<1x4xf16>
    %54 = vector.insert %53, %52 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %39 : !frisk.async_task
    %subview_11 = memref.subview %arg0[0, %17, %10, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %55 = frisk.async_copy_gm_to_shm %subview_11 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = #map2} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %56 = memref.load %view[%19] : memref<4096xf16, 3>
    %57 = memref.load %view[%21] : memref<4096xf16, 3>
    %58 = memref.load %view[%23] : memref<4096xf16, 3>
    %59 = memref.load %view[%25] : memref<4096xf16, 3>
    %60 = vector.from_elements %56, %57, %58, %59 : vector<1x4xf16>
    %61:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %60) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %108 = affine.apply #map13(%arg4)[%thread_id_x]
      %109 = memref.load %view[%108] : memref<4096xf16, 3>
      %110 = affine.apply #map14(%arg4)[%thread_id_x]
      %111 = memref.load %view[%110] : memref<4096xf16, 3>
      %112 = affine.apply #map15(%arg4)[%thread_id_x]
      %113 = memref.load %view[%112] : memref<4096xf16, 3>
      %114 = affine.apply #map16(%arg4)[%thread_id_x]
      %115 = memref.load %view[%114] : memref<4096xf16, 3>
      %116 = vector.from_elements %109, %111, %113, %115 : vector<1x4xf16>
      %117 = affine.apply #map4(%arg4)
      %118 = affine.apply #map3(%arg4)
      %119 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %arg5 [%117, %118] : f16 into vector<2x8xf16>
      %121 = affine.apply #map5(%arg4)
      %122 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %120 [%117, %121] : f16 into vector<2x8xf16>
      %124 = affine.apply #map6(%arg4)
      %125 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %123 [%117, %124] : f16 into vector<2x8xf16>
      %127 = affine.apply #map7(%arg4)
      %128 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %126 [%117, %127] : f16 into vector<2x8xf16>
      %130 = vector.insert %12, %arg6 [%117, %118] : f16 into vector<2x8xf16>
      %131 = vector.insert %12, %130 [%117, %121] : f16 into vector<2x8xf16>
      %132 = vector.insert %12, %131 [%117, %124] : f16 into vector<2x8xf16>
      %133 = vector.insert %12, %132 [%117, %127] : f16 into vector<2x8xf16>
      %134 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%117, %118] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%117, %121] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%117, %124] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%117, %127] : f16 into vector<2x8xf16>
      affine.yield %129, %133, %142, %116 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %62 = arith.mulf %61#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %63 = vector.extract %62[0, 0] : f16 from vector<1x4xf16>
    %64 = vector.insert %63, %61#2 [1, 4] : f16 into vector<2x8xf16>
    %65 = vector.extract %62[0, 1] : f16 from vector<1x4xf16>
    %66 = vector.insert %65, %64 [1, 5] : f16 into vector<2x8xf16>
    %67 = vector.extract %62[0, 2] : f16 from vector<1x4xf16>
    %68 = vector.insert %67, %66 [1, 6] : f16 into vector<2x8xf16>
    %69 = vector.extract %62[0, 3] : f16 from vector<1x4xf16>
    %70 = vector.insert %69, %68 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %55 : !frisk.async_task
    %71 = memref.load %view_8[%19] : memref<4096xf16, 3>
    %72 = memref.load %view_8[%21] : memref<4096xf16, 3>
    %73 = memref.load %view_8[%23] : memref<4096xf16, 3>
    %74 = memref.load %view_8[%25] : memref<4096xf16, 3>
    %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf16>
    %76:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %75) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %108 = affine.apply #map13(%arg4)[%thread_id_x]
      %109 = memref.load %view_8[%108] : memref<4096xf16, 3>
      %110 = affine.apply #map14(%arg4)[%thread_id_x]
      %111 = memref.load %view_8[%110] : memref<4096xf16, 3>
      %112 = affine.apply #map15(%arg4)[%thread_id_x]
      %113 = memref.load %view_8[%112] : memref<4096xf16, 3>
      %114 = affine.apply #map16(%arg4)[%thread_id_x]
      %115 = memref.load %view_8[%114] : memref<4096xf16, 3>
      %116 = vector.from_elements %109, %111, %113, %115 : vector<1x4xf16>
      %117 = affine.apply #map4(%arg4)
      %118 = affine.apply #map3(%arg4)
      %119 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %arg5 [%117, %118] : f16 into vector<2x8xf16>
      %121 = affine.apply #map5(%arg4)
      %122 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %120 [%117, %121] : f16 into vector<2x8xf16>
      %124 = affine.apply #map6(%arg4)
      %125 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %123 [%117, %124] : f16 into vector<2x8xf16>
      %127 = affine.apply #map7(%arg4)
      %128 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %126 [%117, %127] : f16 into vector<2x8xf16>
      %130 = vector.insert %12, %arg6 [%117, %118] : f16 into vector<2x8xf16>
      %131 = vector.insert %12, %130 [%117, %121] : f16 into vector<2x8xf16>
      %132 = vector.insert %12, %131 [%117, %124] : f16 into vector<2x8xf16>
      %133 = vector.insert %12, %132 [%117, %127] : f16 into vector<2x8xf16>
      %134 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%117, %118] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%117, %121] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%117, %124] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%117, %127] : f16 into vector<2x8xf16>
      affine.yield %129, %133, %142, %116 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %77 = arith.mulf %76#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %78 = vector.extract %77[0, 0] : f16 from vector<1x4xf16>
    %79 = vector.insert %78, %76#2 [1, 4] : f16 into vector<2x8xf16>
    %80 = vector.extract %77[0, 1] : f16 from vector<1x4xf16>
    %81 = vector.insert %80, %79 [1, 5] : f16 into vector<2x8xf16>
    %82 = vector.extract %77[0, 2] : f16 from vector<1x4xf16>
    %83 = vector.insert %82, %81 [1, 6] : f16 into vector<2x8xf16>
    %84 = vector.extract %77[0, 3] : f16 from vector<1x4xf16>
    %85 = vector.insert %84, %83 [1, 7] : f16 into vector<2x8xf16>
    gpu.barrier
    %86 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_3) -> (vector<2x32xf32>) {
      %108 = affine.apply #map17(%arg4)
      %109 = affine.apply #map18(%arg4)
      %110 = vector.insert %cst_7, %arg5 [%109, %108] : f32 into vector<2x32xf32>
      %111 = affine.apply #map19(%arg4)
      %112 = vector.insert %cst_7, %110 [%109, %111] : f32 into vector<2x32xf32>
      %113 = affine.apply #map20(%arg4)
      %114 = vector.insert %cst_7, %112 [%109, %113] : f32 into vector<2x32xf32>
      %115 = affine.apply #map21(%arg4)
      %116 = vector.insert %cst_7, %114 [%109, %115] : f32 into vector<2x32xf32>
      affine.yield %116 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
    %87 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_3) -> (vector<2x32xf32>) {
      %108 = affine.apply #map18(%arg4)
      %109 = affine.apply #map17(%arg4)
      %110 = vector.extract %86[%108, %109] : f32 from vector<2x32xf32>
      %111 = affine.apply #map19(%arg4)
      %112 = vector.extract %86[%108, %111] : f32 from vector<2x32xf32>
      %113 = affine.apply #map20(%arg4)
      %114 = vector.extract %86[%108, %113] : f32 from vector<2x32xf32>
      %115 = affine.apply #map21(%arg4)
      %116 = vector.extract %86[%108, %115] : f32 from vector<2x32xf32>
      %117 = vector.insert %110, %arg5 [%108, %109] : f32 into vector<2x32xf32>
      %118 = vector.insert %112, %117 [%108, %111] : f32 into vector<2x32xf32>
      %119 = vector.insert %114, %118 [%108, %113] : f32 into vector<2x32xf32>
      %120 = vector.insert %116, %119 [%108, %115] : f32 into vector<2x32xf32>
      affine.yield %120 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
    %88 = affine.apply #map22()[%thread_id_x]
    %89 = affine.apply #map23()[%thread_id_x]
    %90 = affine.apply #map24()[%thread_id_x]
    %91 = affine.apply #map25()[%thread_id_x]
    %92:6 = affine.for %arg4 = 0 to #map26()[%7] iter_args(%arg5 = %15, %arg6 = %16, %arg7 = %14#3, %arg8 = %15, %arg9 = %15, %arg10 = %15) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %108 = affine.apply #map27(%arg4)[%8, %block_id_z]
      %109 = affine.apply #map28(%arg4)
      %110 = affine.apply #map29(%arg4)
      %subview_12 = memref.subview %transpose[0, %108, %109, %110] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %111 = frisk.async_copy_gm_to_shm %subview_12 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = #map30} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %111 : !frisk.async_task
      %112 = affine.apply #map31(%arg4)[%8, %block_id_z]
      %113 = affine.apply #map32(%arg4)[%8, %block_id_z]
      %subview_13 = memref.subview %transpose[0, %112, %113, %110] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %114 = frisk.async_copy_gm_to_shm %subview_13 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = #map30} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %115:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %87, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %159 = affine.apply #map18(%arg11)
        %160 = affine.apply #map17(%arg11)
        %161 = vector.extract %87[%159, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map19(%arg11)
        %163 = vector.extract %87[%159, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map20(%arg11)
        %165 = vector.extract %87[%159, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map21(%arg11)
        %167 = vector.extract %87[%159, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = affine.apply #map33(%arg11)[%thread_id_x]
        %170 = memref.load %view[%169] : memref<4096xf16, 3>
        %171 = affine.apply #map34(%arg11)[%thread_id_x]
        %172 = memref.load %view[%171] : memref<4096xf16, 3>
        %173 = affine.apply #map35(%arg11)[%thread_id_x]
        %174 = memref.load %view[%173] : memref<4096xf16, 3>
        %175 = affine.apply #map36(%arg11)[%thread_id_x]
        %176 = memref.load %view[%175] : memref<4096xf16, 3>
        %177 = vector.from_elements %170, %172, %174, %176 : vector<4x1xf16>
        %178 = affine.apply #map37(%arg11)[%thread_id_x]
        %179 = memref.load %view[%178] : memref<4096xf16, 3>
        %180 = affine.apply #map38(%arg11)[%thread_id_x]
        %181 = memref.load %view[%180] : memref<4096xf16, 3>
        %182 = affine.apply #map39(%arg11)[%thread_id_x]
        %183 = memref.load %view[%182] : memref<4096xf16, 3>
        %184 = affine.apply #map40(%arg11)[%thread_id_x]
        %185 = memref.load %view[%184] : memref<4096xf16, 3>
        %186 = vector.from_elements %179, %181, %183, %185 : vector<4x1xf16>
        %187 = vector.extract %38[%159, 0] : f16 from vector<2x8xf16>
        %188 = vector.extract %38[%159, 1] : f16 from vector<2x8xf16>
        %189 = vector.extract %38[%159, 2] : f16 from vector<2x8xf16>
        %190 = vector.extract %38[%159, 3] : f16 from vector<2x8xf16>
        %191 = vector.from_elements %187, %188, %189, %190 : vector<1x4xf16>
        %192 = frisk.warp_mma_rr %191, %177, %168 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %193 = vector.extract %38[%159, 4] : f16 from vector<2x8xf16>
        %194 = vector.extract %38[%159, 5] : f16 from vector<2x8xf16>
        %195 = vector.extract %38[%159, 6] : f16 from vector<2x8xf16>
        %196 = vector.extract %38[%159, 7] : f16 from vector<2x8xf16>
        %197 = vector.from_elements %193, %194, %195, %196 : vector<1x4xf16>
        %198 = frisk.warp_mma_rr %197, %186, %192 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %199 = vector.extract %198[0, 0] : f32 from vector<1x4xf32>
        %200 = vector.insert %199, %arg12 [%159, %160] : f32 into vector<2x32xf32>
        %201 = vector.extract %198[0, 1] : f32 from vector<1x4xf32>
        %202 = vector.insert %201, %200 [%159, %162] : f32 into vector<2x32xf32>
        %203 = vector.extract %198[0, 2] : f32 from vector<1x4xf32>
        %204 = vector.insert %203, %202 [%159, %164] : f32 into vector<2x32xf32>
        %205 = vector.extract %198[0, 3] : f32 from vector<1x4xf32>
        %206 = vector.insert %205, %204 [%159, %166] : f32 into vector<2x32xf32>
        %207 = vector.insert %199, %arg13 [%159, %160] : f32 into vector<2x32xf32>
        %208 = vector.insert %201, %207 [%159, %162] : f32 into vector<2x32xf32>
        %209 = vector.insert %203, %208 [%159, %164] : f32 into vector<2x32xf32>
        %210 = vector.insert %205, %209 [%159, %166] : f32 into vector<2x32xf32>
        affine.yield %206, %210 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %116 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %159 = affine.apply #map18(%arg11)
        %160 = affine.apply #map17(%arg11)
        %161 = vector.extract %115#1[%159, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map19(%arg11)
        %163 = vector.extract %115#1[%159, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map20(%arg11)
        %165 = vector.extract %115#1[%159, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map21(%arg11)
        %167 = vector.extract %115#1[%159, %166] : f32 from vector<2x32xf32>
        %168 = vector.insert %161, %arg12 [%159, %160] : f32 into vector<2x32xf32>
        %169 = vector.insert %163, %168 [%159, %162] : f32 into vector<2x32xf32>
        %170 = vector.insert %165, %169 [%159, %164] : f32 into vector<2x32xf32>
        %171 = vector.insert %167, %170 [%159, %166] : f32 into vector<2x32xf32>
        affine.yield %171 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %114 : !frisk.async_task
      %117 = affine.apply #map41(%arg4)[%8, %block_id_z]
      %118 = affine.apply #map42(%arg4)[%8, %block_id_z]
      %subview_14 = memref.subview %transpose[0, %117, %118, %110] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %119 = frisk.async_copy_gm_to_shm %subview_14 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = #map30} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %120:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %116, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %159 = affine.apply #map18(%arg11)
        %160 = affine.apply #map17(%arg11)
        %161 = vector.extract %116[%159, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map19(%arg11)
        %163 = vector.extract %116[%159, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map20(%arg11)
        %165 = vector.extract %116[%159, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map21(%arg11)
        %167 = vector.extract %116[%159, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = affine.apply #map33(%arg11)[%thread_id_x]
        %170 = memref.load %view_8[%169] : memref<4096xf16, 3>
        %171 = affine.apply #map34(%arg11)[%thread_id_x]
        %172 = memref.load %view_8[%171] : memref<4096xf16, 3>
        %173 = affine.apply #map35(%arg11)[%thread_id_x]
        %174 = memref.load %view_8[%173] : memref<4096xf16, 3>
        %175 = affine.apply #map36(%arg11)[%thread_id_x]
        %176 = memref.load %view_8[%175] : memref<4096xf16, 3>
        %177 = vector.from_elements %170, %172, %174, %176 : vector<4x1xf16>
        %178 = affine.apply #map37(%arg11)[%thread_id_x]
        %179 = memref.load %view_8[%178] : memref<4096xf16, 3>
        %180 = affine.apply #map38(%arg11)[%thread_id_x]
        %181 = memref.load %view_8[%180] : memref<4096xf16, 3>
        %182 = affine.apply #map39(%arg11)[%thread_id_x]
        %183 = memref.load %view_8[%182] : memref<4096xf16, 3>
        %184 = affine.apply #map40(%arg11)[%thread_id_x]
        %185 = memref.load %view_8[%184] : memref<4096xf16, 3>
        %186 = vector.from_elements %179, %181, %183, %185 : vector<4x1xf16>
        %187 = vector.extract %54[%159, 0] : f16 from vector<2x8xf16>
        %188 = vector.extract %54[%159, 1] : f16 from vector<2x8xf16>
        %189 = vector.extract %54[%159, 2] : f16 from vector<2x8xf16>
        %190 = vector.extract %54[%159, 3] : f16 from vector<2x8xf16>
        %191 = vector.from_elements %187, %188, %189, %190 : vector<1x4xf16>
        %192 = frisk.warp_mma_rr %191, %177, %168 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %193 = vector.extract %54[%159, 4] : f16 from vector<2x8xf16>
        %194 = vector.extract %54[%159, 5] : f16 from vector<2x8xf16>
        %195 = vector.extract %54[%159, 6] : f16 from vector<2x8xf16>
        %196 = vector.extract %54[%159, 7] : f16 from vector<2x8xf16>
        %197 = vector.from_elements %193, %194, %195, %196 : vector<1x4xf16>
        %198 = frisk.warp_mma_rr %197, %186, %192 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %199 = vector.extract %198[0, 0] : f32 from vector<1x4xf32>
        %200 = vector.insert %199, %arg12 [%159, %160] : f32 into vector<2x32xf32>
        %201 = vector.extract %198[0, 1] : f32 from vector<1x4xf32>
        %202 = vector.insert %201, %200 [%159, %162] : f32 into vector<2x32xf32>
        %203 = vector.extract %198[0, 2] : f32 from vector<1x4xf32>
        %204 = vector.insert %203, %202 [%159, %164] : f32 into vector<2x32xf32>
        %205 = vector.extract %198[0, 3] : f32 from vector<1x4xf32>
        %206 = vector.insert %205, %204 [%159, %166] : f32 into vector<2x32xf32>
        %207 = vector.insert %199, %arg13 [%159, %160] : f32 into vector<2x32xf32>
        %208 = vector.insert %201, %207 [%159, %162] : f32 into vector<2x32xf32>
        %209 = vector.insert %203, %208 [%159, %164] : f32 into vector<2x32xf32>
        %210 = vector.insert %205, %209 [%159, %166] : f32 into vector<2x32xf32>
        affine.yield %206, %210 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %121 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %159 = affine.apply #map18(%arg11)
        %160 = affine.apply #map17(%arg11)
        %161 = vector.extract %120#1[%159, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map19(%arg11)
        %163 = vector.extract %120#1[%159, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map20(%arg11)
        %165 = vector.extract %120#1[%159, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map21(%arg11)
        %167 = vector.extract %120#1[%159, %166] : f32 from vector<2x32xf32>
        %168 = vector.insert %161, %arg12 [%159, %160] : f32 into vector<2x32xf32>
        %169 = vector.insert %163, %168 [%159, %162] : f32 into vector<2x32xf32>
        %170 = vector.insert %165, %169 [%159, %164] : f32 into vector<2x32xf32>
        %171 = vector.insert %167, %170 [%159, %166] : f32 into vector<2x32xf32>
        affine.yield %171 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %119 : !frisk.async_task
      %122 = affine.apply #map43(%arg4)[%8, %block_id_z]
      %123 = affine.apply #map44(%arg4)[%8, %block_id_z]
      %subview_15 = memref.subview %transpose[0, %122, %123, %110] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %124 = frisk.async_copy_gm_to_shm %subview_15 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = #map30} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %125:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %121, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %159 = affine.apply #map18(%arg11)
        %160 = affine.apply #map17(%arg11)
        %161 = vector.extract %121[%159, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map19(%arg11)
        %163 = vector.extract %121[%159, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map20(%arg11)
        %165 = vector.extract %121[%159, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map21(%arg11)
        %167 = vector.extract %121[%159, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = affine.apply #map33(%arg11)[%thread_id_x]
        %170 = memref.load %view[%169] : memref<4096xf16, 3>
        %171 = affine.apply #map34(%arg11)[%thread_id_x]
        %172 = memref.load %view[%171] : memref<4096xf16, 3>
        %173 = affine.apply #map35(%arg11)[%thread_id_x]
        %174 = memref.load %view[%173] : memref<4096xf16, 3>
        %175 = affine.apply #map36(%arg11)[%thread_id_x]
        %176 = memref.load %view[%175] : memref<4096xf16, 3>
        %177 = vector.from_elements %170, %172, %174, %176 : vector<4x1xf16>
        %178 = affine.apply #map37(%arg11)[%thread_id_x]
        %179 = memref.load %view[%178] : memref<4096xf16, 3>
        %180 = affine.apply #map38(%arg11)[%thread_id_x]
        %181 = memref.load %view[%180] : memref<4096xf16, 3>
        %182 = affine.apply #map39(%arg11)[%thread_id_x]
        %183 = memref.load %view[%182] : memref<4096xf16, 3>
        %184 = affine.apply #map40(%arg11)[%thread_id_x]
        %185 = memref.load %view[%184] : memref<4096xf16, 3>
        %186 = vector.from_elements %179, %181, %183, %185 : vector<4x1xf16>
        %187 = vector.extract %70[%159, 0] : f16 from vector<2x8xf16>
        %188 = vector.extract %70[%159, 1] : f16 from vector<2x8xf16>
        %189 = vector.extract %70[%159, 2] : f16 from vector<2x8xf16>
        %190 = vector.extract %70[%159, 3] : f16 from vector<2x8xf16>
        %191 = vector.from_elements %187, %188, %189, %190 : vector<1x4xf16>
        %192 = frisk.warp_mma_rr %191, %177, %168 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %193 = vector.extract %70[%159, 4] : f16 from vector<2x8xf16>
        %194 = vector.extract %70[%159, 5] : f16 from vector<2x8xf16>
        %195 = vector.extract %70[%159, 6] : f16 from vector<2x8xf16>
        %196 = vector.extract %70[%159, 7] : f16 from vector<2x8xf16>
        %197 = vector.from_elements %193, %194, %195, %196 : vector<1x4xf16>
        %198 = frisk.warp_mma_rr %197, %186, %192 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %199 = vector.extract %198[0, 0] : f32 from vector<1x4xf32>
        %200 = vector.insert %199, %arg12 [%159, %160] : f32 into vector<2x32xf32>
        %201 = vector.extract %198[0, 1] : f32 from vector<1x4xf32>
        %202 = vector.insert %201, %200 [%159, %162] : f32 into vector<2x32xf32>
        %203 = vector.extract %198[0, 2] : f32 from vector<1x4xf32>
        %204 = vector.insert %203, %202 [%159, %164] : f32 into vector<2x32xf32>
        %205 = vector.extract %198[0, 3] : f32 from vector<1x4xf32>
        %206 = vector.insert %205, %204 [%159, %166] : f32 into vector<2x32xf32>
        %207 = vector.insert %199, %arg13 [%159, %160] : f32 into vector<2x32xf32>
        %208 = vector.insert %201, %207 [%159, %162] : f32 into vector<2x32xf32>
        %209 = vector.insert %203, %208 [%159, %164] : f32 into vector<2x32xf32>
        %210 = vector.insert %205, %209 [%159, %166] : f32 into vector<2x32xf32>
        affine.yield %206, %210 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %126 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %159 = affine.apply #map18(%arg11)
        %160 = affine.apply #map17(%arg11)
        %161 = vector.extract %125#1[%159, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map19(%arg11)
        %163 = vector.extract %125#1[%159, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map20(%arg11)
        %165 = vector.extract %125#1[%159, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map21(%arg11)
        %167 = vector.extract %125#1[%159, %166] : f32 from vector<2x32xf32>
        %168 = vector.insert %161, %arg12 [%159, %160] : f32 into vector<2x32xf32>
        %169 = vector.insert %163, %168 [%159, %162] : f32 into vector<2x32xf32>
        %170 = vector.insert %165, %169 [%159, %164] : f32 into vector<2x32xf32>
        %171 = vector.insert %167, %170 [%159, %166] : f32 into vector<2x32xf32>
        affine.yield %171 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %124 : !frisk.async_task
      %127 = affine.apply #map45(%arg4)[%8, %block_id_z]
      %subview_16 = memref.subview %arg2[0, %127, %110, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %128 = frisk.async_copy_gm_to_shm %subview_16 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = #map46} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %129:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %126, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %159 = affine.apply #map18(%arg11)
        %160 = affine.apply #map17(%arg11)
        %161 = vector.extract %126[%159, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map19(%arg11)
        %163 = vector.extract %126[%159, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map20(%arg11)
        %165 = vector.extract %126[%159, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map21(%arg11)
        %167 = vector.extract %126[%159, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = affine.apply #map33(%arg11)[%thread_id_x]
        %170 = memref.load %view_8[%169] : memref<4096xf16, 3>
        %171 = affine.apply #map34(%arg11)[%thread_id_x]
        %172 = memref.load %view_8[%171] : memref<4096xf16, 3>
        %173 = affine.apply #map35(%arg11)[%thread_id_x]
        %174 = memref.load %view_8[%173] : memref<4096xf16, 3>
        %175 = affine.apply #map36(%arg11)[%thread_id_x]
        %176 = memref.load %view_8[%175] : memref<4096xf16, 3>
        %177 = vector.from_elements %170, %172, %174, %176 : vector<4x1xf16>
        %178 = affine.apply #map37(%arg11)[%thread_id_x]
        %179 = memref.load %view_8[%178] : memref<4096xf16, 3>
        %180 = affine.apply #map38(%arg11)[%thread_id_x]
        %181 = memref.load %view_8[%180] : memref<4096xf16, 3>
        %182 = affine.apply #map39(%arg11)[%thread_id_x]
        %183 = memref.load %view_8[%182] : memref<4096xf16, 3>
        %184 = affine.apply #map40(%arg11)[%thread_id_x]
        %185 = memref.load %view_8[%184] : memref<4096xf16, 3>
        %186 = vector.from_elements %179, %181, %183, %185 : vector<4x1xf16>
        %187 = vector.extract %85[%159, 0] : f16 from vector<2x8xf16>
        %188 = vector.extract %85[%159, 1] : f16 from vector<2x8xf16>
        %189 = vector.extract %85[%159, 2] : f16 from vector<2x8xf16>
        %190 = vector.extract %85[%159, 3] : f16 from vector<2x8xf16>
        %191 = vector.from_elements %187, %188, %189, %190 : vector<1x4xf16>
        %192 = frisk.warp_mma_rr %191, %177, %168 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %193 = vector.extract %85[%159, 4] : f16 from vector<2x8xf16>
        %194 = vector.extract %85[%159, 5] : f16 from vector<2x8xf16>
        %195 = vector.extract %85[%159, 6] : f16 from vector<2x8xf16>
        %196 = vector.extract %85[%159, 7] : f16 from vector<2x8xf16>
        %197 = vector.from_elements %193, %194, %195, %196 : vector<1x4xf16>
        %198 = frisk.warp_mma_rr %197, %186, %192 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %199 = vector.extract %198[0, 0] : f32 from vector<1x4xf32>
        %200 = vector.insert %199, %arg12 [%159, %160] : f32 into vector<2x32xf32>
        %201 = vector.extract %198[0, 1] : f32 from vector<1x4xf32>
        %202 = vector.insert %201, %200 [%159, %162] : f32 into vector<2x32xf32>
        %203 = vector.extract %198[0, 2] : f32 from vector<1x4xf32>
        %204 = vector.insert %203, %202 [%159, %164] : f32 into vector<2x32xf32>
        %205 = vector.extract %198[0, 3] : f32 from vector<1x4xf32>
        %206 = vector.insert %205, %204 [%159, %166] : f32 into vector<2x32xf32>
        %207 = vector.insert %199, %arg13 [%159, %160] : f32 into vector<2x32xf32>
        %208 = vector.insert %201, %207 [%159, %162] : f32 into vector<2x32xf32>
        %209 = vector.insert %203, %208 [%159, %164] : f32 into vector<2x32xf32>
        %210 = vector.insert %205, %209 [%159, %166] : f32 into vector<2x32xf32>
        affine.yield %206, %210 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %130 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %159 = affine.apply #map17(%arg11)
        %160 = affine.apply #map18(%arg11)
        %161 = affine.apply #map47(%arg11)[%thread_id_x, %13]
        %162 = affine.apply #map48(%arg11, %arg4)[%thread_id_x]
        %163 = arith.index_cast %161 : index to i32
        %164 = arith.index_cast %162 : index to i32
        %165 = arith.addi %163, %c1_i32 : i32
        %166 = arith.cmpi ule, %165, %164 {frisk.causal_mask_i32} : i32
        %167 = vector.extract %129#1[%160, %159] : f32 from vector<2x32xf32>
        %168 = arith.select %166, %cst_0, %167 {frisk.causal_mask_select} : f32
        %169 = affine.apply #map19(%arg11)
        %170 = affine.apply #map49(%arg11, %arg4)[%thread_id_x]
        %171 = arith.index_cast %170 : index to i32
        %172 = arith.cmpi ule, %165, %171 {frisk.causal_mask_i32} : i32
        %173 = vector.extract %129#1[%160, %169] : f32 from vector<2x32xf32>
        %174 = arith.select %172, %cst_0, %173 {frisk.causal_mask_select} : f32
        %175 = affine.apply #map20(%arg11)
        %176 = affine.apply #map50(%arg11, %arg4)[%thread_id_x]
        %177 = arith.index_cast %176 : index to i32
        %178 = arith.cmpi ule, %165, %177 {frisk.causal_mask_i32} : i32
        %179 = vector.extract %129#1[%160, %175] : f32 from vector<2x32xf32>
        %180 = arith.select %178, %cst_0, %179 {frisk.causal_mask_select} : f32
        %181 = affine.apply #map21(%arg11)
        %182 = affine.apply #map51(%arg11, %arg4)[%thread_id_x]
        %183 = arith.index_cast %182 : index to i32
        %184 = arith.cmpi ule, %165, %183 {frisk.causal_mask_i32} : i32
        %185 = vector.extract %129#1[%160, %181] : f32 from vector<2x32xf32>
        %186 = arith.select %184, %cst_0, %185 {frisk.causal_mask_select} : f32
        %187 = vector.insert %168, %arg12 [%160, %159] : f32 into vector<2x32xf32>
        %188 = vector.insert %174, %187 [%160, %169] : f32 into vector<2x32xf32>
        %189 = vector.insert %180, %188 [%160, %175] : f32 into vector<2x32xf32>
        %190 = vector.insert %186, %189 [%160, %181] : f32 into vector<2x32xf32>
        affine.yield %190 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "mask_fragment", tiled = true}
      %131 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %159 = affine.apply #map18(%arg11)
        %160 = affine.apply #map17(%arg11)
        %161 = vector.extract %130[%159, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map19(%arg11)
        %163 = vector.extract %130[%159, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map20(%arg11)
        %165 = vector.extract %130[%159, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map21(%arg11)
        %167 = vector.extract %130[%159, %166] : f32 from vector<2x32xf32>
        %168 = vector.insert %161, %arg12 [%159, %160] : f32 into vector<2x32xf32>
        %169 = vector.insert %163, %168 [%159, %162] : f32 into vector<2x32xf32>
        %170 = vector.insert %165, %169 [%159, %164] : f32 into vector<2x32xf32>
        %171 = vector.insert %167, %170 [%159, %166] : f32 into vector<2x32xf32>
        affine.yield %171 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %132 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %159 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_0) -> (f32) {
          %163 = affine.apply #map52(%arg13)
          %164 = vector.extract %131[%arg11, %163] : f32 from vector<2x32xf32>
          %165 = arith.maxnumf %arg14, %164 {frisk.try_multichain} : f32
          %166 = affine.apply #map53(%arg13)
          %167 = vector.extract %131[%arg11, %166] : f32 from vector<2x32xf32>
          %168 = arith.maxnumf %165, %167 {frisk.try_multichain} : f32
          %169 = affine.apply #map54(%arg13)
          %170 = vector.extract %131[%arg11, %169] : f32 from vector<2x32xf32>
          %171 = arith.maxnumf %168, %170 {frisk.try_multichain} : f32
          %172 = affine.apply #map55(%arg13)
          %173 = vector.extract %131[%arg11, %172] : f32 from vector<2x32xf32>
          %174 = arith.maxnumf %171, %173 {frisk.try_multichain} : f32
          affine.yield %174 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %159, %c16_i32, %c64_i32 : f32
        %160 = arith.maxnumf %159, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %160, %c32_i32, %c64_i32 : f32
        %161 = arith.maxnumf %160, %shuffleResult_31 : f32
        %162 = vector.insert %161, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %162 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %133 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %159 = vector.extract %132[%arg11, 0] : f32 from vector<2x1xf32>
        %160 = vector.broadcast %159 : f32 to vector<1x1xf32>
        %161 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %162 = vector.broadcast %161 : f32 to vector<1x1xf32>
        %163 = arith.maxnumf %160, %162 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %164 = vector.extract %163[0, 0] : f32 from vector<1x1xf32>
        %165 = vector.insert %164, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %165 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %134:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %159 = affine.apply #map18(%arg11)
        %160 = vector.extract %133[%159, 0] : f32 from vector<2x1xf32>
        %161 = vector.broadcast %160 : f32 to vector<1x4xf32>
        %162 = affine.apply #map17(%arg11)
        %163 = vector.insert %160, %arg12 [%159, %162] : f32 into vector<2x32xf32>
        %164 = affine.apply #map19(%arg11)
        %165 = vector.insert %160, %163 [%159, %164] : f32 into vector<2x32xf32>
        %166 = affine.apply #map20(%arg11)
        %167 = vector.insert %160, %165 [%159, %166] : f32 into vector<2x32xf32>
        %168 = affine.apply #map21(%arg11)
        %169 = vector.insert %160, %167 [%159, %168] : f32 into vector<2x32xf32>
        %170 = vector.extract %130[%159, %162] : f32 from vector<2x32xf32>
        %171 = vector.extract %130[%159, %164] : f32 from vector<2x32xf32>
        %172 = vector.extract %130[%159, %166] : f32 from vector<2x32xf32>
        %173 = vector.extract %130[%159, %168] : f32 from vector<2x32xf32>
        %174 = vector.from_elements %170, %171, %172, %173 : vector<1x4xf32>
        %175 = arith.subf %174, %161 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %176 = vector.extract %175[0, 0] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %arg13 [%159, %162] : f32 into vector<2x32xf32>
        %178 = vector.extract %175[0, 1] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%159, %164] : f32 into vector<2x32xf32>
        %180 = vector.extract %175[0, 2] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%159, %166] : f32 into vector<2x32xf32>
        %182 = vector.extract %175[0, 3] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %181 [%159, %168] : f32 into vector<2x32xf32>
        affine.yield %169, %183 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %135:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1, %arg14 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %159 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %160 = vector.broadcast %159 : f32 to vector<1x1xf32>
        %161 = vector.extract %133[%arg11, 0] : f32 from vector<2x1xf32>
        %162 = vector.broadcast %161 : f32 to vector<1x1xf32>
        %163 = arith.subf %160, %162 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %164 = vector.extract %163[0, 0] : f32 from vector<1x1xf32>
        %165 = vector.insert %164, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %166 = math.exp2 %163 {tiled = true} : vector<1x1xf32>
        %167 = vector.extract %166[0, 0] : f32 from vector<1x1xf32>
        %168 = vector.insert %167, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %169 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %170 = vector.broadcast %169 : f32 to vector<1x1xf32>
        %171 = arith.mulf %170, %166 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %172 = vector.extract %171[0, 0] : f32 from vector<1x1xf32>
        %173 = vector.insert %172, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %165, %168, %173 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %136:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %159 = affine.apply #map18(%arg11)
        %160 = affine.apply #map17(%arg11)
        %161 = vector.extract %134#1[%159, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map19(%arg11)
        %163 = vector.extract %134#1[%159, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map20(%arg11)
        %165 = vector.extract %134#1[%159, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map21(%arg11)
        %167 = vector.extract %134#1[%159, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = math.exp2 %168 {tiled = true} : vector<1x4xf32>
        %170 = vector.extract %169[0, 0] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %arg12 [%159, %160] : f32 into vector<2x32xf32>
        %172 = vector.extract %169[0, 1] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%159, %162] : f32 into vector<2x32xf32>
        %174 = vector.extract %169[0, 2] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%159, %164] : f32 into vector<2x32xf32>
        %176 = vector.extract %169[0, 3] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%159, %166] : f32 into vector<2x32xf32>
        %178 = vector.insert %170, %arg13 [%159, %160] : f32 into vector<2x32xf32>
        %179 = vector.insert %172, %178 [%159, %162] : f32 into vector<2x32xf32>
        %180 = vector.insert %174, %179 [%159, %164] : f32 into vector<2x32xf32>
        %181 = vector.insert %176, %180 [%159, %166] : f32 into vector<2x32xf32>
        affine.yield %177, %181 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %137 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %159 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_7) -> (f32) {
          %163 = affine.apply #map52(%arg13)
          %164 = vector.extract %136#1[%arg11, %163] : f32 from vector<2x32xf32>
          %165 = arith.addf %arg14, %164 {frisk.try_multichain} : f32
          %166 = affine.apply #map53(%arg13)
          %167 = vector.extract %136#1[%arg11, %166] : f32 from vector<2x32xf32>
          %168 = arith.addf %165, %167 {frisk.try_multichain} : f32
          %169 = affine.apply #map54(%arg13)
          %170 = vector.extract %136#1[%arg11, %169] : f32 from vector<2x32xf32>
          %171 = arith.addf %168, %170 {frisk.try_multichain} : f32
          %172 = affine.apply #map55(%arg13)
          %173 = vector.extract %136#1[%arg11, %172] : f32 from vector<2x32xf32>
          %174 = arith.addf %171, %173 {frisk.try_multichain} : f32
          affine.yield %174 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %159, %c16_i32, %c64_i32 : f32
        %160 = arith.addf %159, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %160, %c32_i32, %c64_i32 : f32
        %161 = arith.addf %160, %shuffleResult_31 : f32
        %162 = vector.insert %161, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %162 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %138 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %159 = vector.extract %137[%arg11, 0] : f32 from vector<2x1xf32>
        %160 = vector.broadcast %159 : f32 to vector<1x1xf32>
        %161 = vector.extract %135#2[%arg11, 0] : f32 from vector<2x1xf32>
        %162 = vector.broadcast %161 : f32 to vector<1x1xf32>
        %163 = arith.addf %160, %162 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %164 = vector.extract %163[0, 0] : f32 from vector<1x1xf32>
        %165 = vector.insert %164, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %165 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %139 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf16>) {
        %159 = affine.apply #map18(%arg11)
        %160 = affine.apply #map17(%arg11)
        %161 = vector.extract %136#0[%159, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map19(%arg11)
        %163 = vector.extract %136#0[%159, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map20(%arg11)
        %165 = vector.extract %136#0[%159, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map21(%arg11)
        %167 = vector.extract %136#0[%159, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = arith.truncf %168 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %170 = vector.extract %169[0, 0] : f16 from vector<1x4xf16>
        %171 = vector.insert %170, %arg12 [%159, %160] : f16 into vector<2x32xf16>
        %172 = vector.extract %169[0, 1] : f16 from vector<1x4xf16>
        %173 = vector.insert %172, %171 [%159, %162] : f16 into vector<2x32xf16>
        %174 = vector.extract %169[0, 2] : f16 from vector<1x4xf16>
        %175 = vector.insert %174, %173 [%159, %164] : f16 into vector<2x32xf16>
        %176 = vector.extract %169[0, 3] : f16 from vector<1x4xf16>
        %177 = vector.insert %176, %175 [%159, %166] : f16 into vector<2x32xf16>
        affine.yield %177 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %140:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %159 = affine.apply #map4(%arg11)
        %160 = vector.extract %135#1[%159, 0] : f32 from vector<2x1xf32>
        %161 = vector.broadcast %160 : f32 to vector<1x4xf32>
        %162 = affine.apply #map3(%arg11)
        %163 = vector.insert %160, %arg12 [%159, %162] : f32 into vector<2x8xf32>
        %164 = affine.apply #map5(%arg11)
        %165 = vector.insert %160, %163 [%159, %164] : f32 into vector<2x8xf32>
        %166 = affine.apply #map6(%arg11)
        %167 = vector.insert %160, %165 [%159, %166] : f32 into vector<2x8xf32>
        %168 = affine.apply #map7(%arg11)
        %169 = vector.insert %160, %167 [%159, %168] : f32 into vector<2x8xf32>
        %170 = vector.extract %arg5[%159, %162] : f32 from vector<2x8xf32>
        %171 = vector.extract %arg5[%159, %164] : f32 from vector<2x8xf32>
        %172 = vector.extract %arg5[%159, %166] : f32 from vector<2x8xf32>
        %173 = vector.extract %arg5[%159, %168] : f32 from vector<2x8xf32>
        %174 = vector.from_elements %170, %171, %172, %173 : vector<1x4xf32>
        %175 = arith.mulf %174, %161 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %176 = vector.extract %175[0, 0] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %arg13 [%159, %162] : f32 into vector<2x8xf32>
        %178 = vector.extract %175[0, 1] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%159, %164] : f32 into vector<2x8xf32>
        %180 = vector.extract %175[0, 2] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%159, %166] : f32 into vector<2x8xf32>
        %182 = vector.extract %175[0, 3] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %181 [%159, %168] : f32 into vector<2x8xf32>
        affine.yield %169, %183 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %128 : !frisk.async_task
      %141 = affine.apply #map56(%arg4)[%8]
      %subview_17 = memref.subview %arg2[0, %141, %110, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %c8192_18 = arith.constant 8192 : index
      %view_19 = memref.view %alloc[%c8192_18][] : memref<16384xi8, 3> to memref<4096xf16, 3>
      %c8192_20 = arith.constant 8192 : index
      %view_21 = memref.view %alloc[%c8192_20][] : memref<16384xi8, 3> to memref<4096xf16, 3>
      %142 = frisk.async_copy_gm_to_shm %subview_17 to %view_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = #map46} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %143 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %159 = affine.apply #map57(%arg11)[%thread_id_x]
        %160 = frisk.read_matrix_from_shm %view[%159] {element_bit_width = 16 : i64, frisk.v_double_buffer = 8192 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %161 = vector.bitcast %160 : vector<4xi32> to vector<8xf16>
        %162 = vector.extract %161[0] : f16 from vector<8xf16>
        %163 = vector.extract %161[%88] : f16 from vector<8xf16>
        %164 = vector.extract %161[1] : f16 from vector<8xf16>
        %165 = vector.extract %161[%89] : f16 from vector<8xf16>
        %166 = vector.extract %161[2] : f16 from vector<8xf16>
        %167 = vector.extract %161[%90] : f16 from vector<8xf16>
        %168 = vector.extract %161[3] : f16 from vector<8xf16>
        %169 = vector.extract %161[%91] : f16 from vector<8xf16>
        %170 = affine.apply #map52(%arg11)
        %171 = vector.insert %162, %arg12 [%170, 0] : f16 into vector<32x2xf16>
        %172 = vector.insert %163, %171 [%170, 1] : f16 into vector<32x2xf16>
        %173 = affine.apply #map53(%arg11)
        %174 = vector.insert %164, %172 [%173, 0] : f16 into vector<32x2xf16>
        %175 = vector.insert %165, %174 [%173, 1] : f16 into vector<32x2xf16>
        %176 = affine.apply #map54(%arg11)
        %177 = vector.insert %166, %175 [%176, 0] : f16 into vector<32x2xf16>
        %178 = vector.insert %167, %177 [%176, 1] : f16 into vector<32x2xf16>
        %179 = affine.apply #map55(%arg11)
        %180 = vector.insert %168, %178 [%179, 0] : f16 into vector<32x2xf16>
        %181 = vector.insert %169, %180 [%179, 1] : f16 into vector<32x2xf16>
        affine.yield %181 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %144 = affine.apply #map56(%arg4)[%8]
      %subview_22 = memref.subview %arg2[0, %144, %110, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %145 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %140#1) -> (vector<2x8xf32>) {
        %159 = affine.apply #map4(%arg11)
        %160 = affine.apply #map3(%arg11)
        %161 = vector.extract %140#1[%159, %160] : f32 from vector<2x8xf32>
        %162 = affine.apply #map5(%arg11)
        %163 = vector.extract %140#1[%159, %162] : f32 from vector<2x8xf32>
        %164 = affine.apply #map6(%arg11)
        %165 = vector.extract %140#1[%159, %164] : f32 from vector<2x8xf32>
        %166 = affine.apply #map7(%arg11)
        %167 = vector.extract %140#1[%159, %166] : f32 from vector<2x8xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = vector.extract %139[%159, 0] : f16 from vector<2x32xf16>
        %170 = vector.extract %139[%159, 1] : f16 from vector<2x32xf16>
        %171 = vector.extract %139[%159, 2] : f16 from vector<2x32xf16>
        %172 = vector.extract %139[%159, 3] : f16 from vector<2x32xf16>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf16>
        %174 = affine.apply #map58(%arg11)
        %175 = vector.extract %143[0, %174] : f16 from vector<32x2xf16>
        %176 = vector.extract %143[1, %174] : f16 from vector<32x2xf16>
        %177 = vector.extract %143[2, %174] : f16 from vector<32x2xf16>
        %178 = vector.extract %143[3, %174] : f16 from vector<32x2xf16>
        %179 = vector.from_elements %175, %176, %177, %178 : vector<4x1xf16>
        %180 = frisk.warp_mma_rr %173, %179, %168 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %181 = vector.extract %139[%159, 4] : f16 from vector<2x32xf16>
        %182 = vector.extract %139[%159, 5] : f16 from vector<2x32xf16>
        %183 = vector.extract %139[%159, 6] : f16 from vector<2x32xf16>
        %184 = vector.extract %139[%159, 7] : f16 from vector<2x32xf16>
        %185 = vector.from_elements %181, %182, %183, %184 : vector<1x4xf16>
        %186 = vector.extract %143[4, %174] : f16 from vector<32x2xf16>
        %187 = vector.extract %143[5, %174] : f16 from vector<32x2xf16>
        %188 = vector.extract %143[6, %174] : f16 from vector<32x2xf16>
        %189 = vector.extract %143[7, %174] : f16 from vector<32x2xf16>
        %190 = vector.from_elements %186, %187, %188, %189 : vector<4x1xf16>
        %191 = frisk.warp_mma_rr %185, %190, %180 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %192 = vector.extract %139[%159, 8] : f16 from vector<2x32xf16>
        %193 = vector.extract %139[%159, 9] : f16 from vector<2x32xf16>
        %194 = vector.extract %139[%159, 10] : f16 from vector<2x32xf16>
        %195 = vector.extract %139[%159, 11] : f16 from vector<2x32xf16>
        %196 = vector.from_elements %192, %193, %194, %195 : vector<1x4xf16>
        %197 = vector.extract %143[8, %174] : f16 from vector<32x2xf16>
        %198 = vector.extract %143[9, %174] : f16 from vector<32x2xf16>
        %199 = vector.extract %143[10, %174] : f16 from vector<32x2xf16>
        %200 = vector.extract %143[11, %174] : f16 from vector<32x2xf16>
        %201 = vector.from_elements %197, %198, %199, %200 : vector<4x1xf16>
        %202 = frisk.warp_mma_rr %196, %201, %191 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %203 = vector.extract %139[%159, 12] : f16 from vector<2x32xf16>
        %204 = vector.extract %139[%159, 13] : f16 from vector<2x32xf16>
        %205 = vector.extract %139[%159, 14] : f16 from vector<2x32xf16>
        %206 = vector.extract %139[%159, 15] : f16 from vector<2x32xf16>
        %207 = vector.from_elements %203, %204, %205, %206 : vector<1x4xf16>
        %208 = vector.extract %143[12, %174] : f16 from vector<32x2xf16>
        %209 = vector.extract %143[13, %174] : f16 from vector<32x2xf16>
        %210 = vector.extract %143[14, %174] : f16 from vector<32x2xf16>
        %211 = vector.extract %143[15, %174] : f16 from vector<32x2xf16>
        %212 = vector.from_elements %208, %209, %210, %211 : vector<4x1xf16>
        %213 = frisk.warp_mma_rr %207, %212, %202 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %214 = vector.extract %139[%159, 16] : f16 from vector<2x32xf16>
        %215 = vector.extract %139[%159, 17] : f16 from vector<2x32xf16>
        %216 = vector.extract %139[%159, 18] : f16 from vector<2x32xf16>
        %217 = vector.extract %139[%159, 19] : f16 from vector<2x32xf16>
        %218 = vector.from_elements %214, %215, %216, %217 : vector<1x4xf16>
        %219 = vector.extract %143[16, %174] : f16 from vector<32x2xf16>
        %220 = vector.extract %143[17, %174] : f16 from vector<32x2xf16>
        %221 = vector.extract %143[18, %174] : f16 from vector<32x2xf16>
        %222 = vector.extract %143[19, %174] : f16 from vector<32x2xf16>
        %223 = vector.from_elements %219, %220, %221, %222 : vector<4x1xf16>
        %224 = frisk.warp_mma_rr %218, %223, %213 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %225 = vector.extract %139[%159, 20] : f16 from vector<2x32xf16>
        %226 = vector.extract %139[%159, 21] : f16 from vector<2x32xf16>
        %227 = vector.extract %139[%159, 22] : f16 from vector<2x32xf16>
        %228 = vector.extract %139[%159, 23] : f16 from vector<2x32xf16>
        %229 = vector.from_elements %225, %226, %227, %228 : vector<1x4xf16>
        %230 = vector.extract %143[20, %174] : f16 from vector<32x2xf16>
        %231 = vector.extract %143[21, %174] : f16 from vector<32x2xf16>
        %232 = vector.extract %143[22, %174] : f16 from vector<32x2xf16>
        %233 = vector.extract %143[23, %174] : f16 from vector<32x2xf16>
        %234 = vector.from_elements %230, %231, %232, %233 : vector<4x1xf16>
        %235 = frisk.warp_mma_rr %229, %234, %224 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %236 = vector.extract %139[%159, 24] : f16 from vector<2x32xf16>
        %237 = vector.extract %139[%159, 25] : f16 from vector<2x32xf16>
        %238 = vector.extract %139[%159, 26] : f16 from vector<2x32xf16>
        %239 = vector.extract %139[%159, 27] : f16 from vector<2x32xf16>
        %240 = vector.from_elements %236, %237, %238, %239 : vector<1x4xf16>
        %241 = vector.extract %143[24, %174] : f16 from vector<32x2xf16>
        %242 = vector.extract %143[25, %174] : f16 from vector<32x2xf16>
        %243 = vector.extract %143[26, %174] : f16 from vector<32x2xf16>
        %244 = vector.extract %143[27, %174] : f16 from vector<32x2xf16>
        %245 = vector.from_elements %241, %242, %243, %244 : vector<4x1xf16>
        %246 = frisk.warp_mma_rr %240, %245, %235 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %247 = vector.extract %139[%159, 28] : f16 from vector<2x32xf16>
        %248 = vector.extract %139[%159, 29] : f16 from vector<2x32xf16>
        %249 = vector.extract %139[%159, 30] : f16 from vector<2x32xf16>
        %250 = vector.extract %139[%159, 31] : f16 from vector<2x32xf16>
        %251 = vector.from_elements %247, %248, %249, %250 : vector<1x4xf16>
        %252 = vector.extract %143[28, %174] : f16 from vector<32x2xf16>
        %253 = vector.extract %143[29, %174] : f16 from vector<32x2xf16>
        %254 = vector.extract %143[30, %174] : f16 from vector<32x2xf16>
        %255 = vector.extract %143[31, %174] : f16 from vector<32x2xf16>
        %256 = vector.from_elements %252, %253, %254, %255 : vector<4x1xf16>
        %257 = frisk.warp_mma_rr %251, %256, %246 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %258 = vector.extract %257[0, 0] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %arg12 [%159, %160] : f32 into vector<2x8xf32>
        %260 = vector.extract %257[0, 1] : f32 from vector<1x4xf32>
        %261 = vector.insert %260, %259 [%159, %162] : f32 into vector<2x8xf32>
        %262 = vector.extract %257[0, 2] : f32 from vector<1x4xf32>
        %263 = vector.insert %262, %261 [%159, %164] : f32 into vector<2x8xf32>
        %264 = vector.extract %257[0, 3] : f32 from vector<1x4xf32>
        %265 = vector.insert %264, %263 [%159, %166] : f32 into vector<2x8xf32>
        affine.yield %265 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %146:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %159 = affine.apply #map4(%arg11)
        %160 = vector.extract %135#1[%159, 0] : f32 from vector<2x1xf32>
        %161 = vector.broadcast %160 : f32 to vector<1x4xf32>
        %162 = affine.apply #map3(%arg11)
        %163 = vector.insert %160, %arg12 [%159, %162] : f32 into vector<2x8xf32>
        %164 = affine.apply #map5(%arg11)
        %165 = vector.insert %160, %163 [%159, %164] : f32 into vector<2x8xf32>
        %166 = affine.apply #map6(%arg11)
        %167 = vector.insert %160, %165 [%159, %166] : f32 into vector<2x8xf32>
        %168 = affine.apply #map7(%arg11)
        %169 = vector.insert %160, %167 [%159, %168] : f32 into vector<2x8xf32>
        %170 = vector.extract %arg8[%159, %162] : f32 from vector<2x8xf32>
        %171 = vector.extract %arg8[%159, %164] : f32 from vector<2x8xf32>
        %172 = vector.extract %arg8[%159, %166] : f32 from vector<2x8xf32>
        %173 = vector.extract %arg8[%159, %168] : f32 from vector<2x8xf32>
        %174 = vector.from_elements %170, %171, %172, %173 : vector<1x4xf32>
        %175 = arith.mulf %174, %161 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %176 = vector.extract %175[0, 0] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %arg13 [%159, %162] : f32 into vector<2x8xf32>
        %178 = vector.extract %175[0, 1] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%159, %164] : f32 into vector<2x8xf32>
        %180 = vector.extract %175[0, 2] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%159, %166] : f32 into vector<2x8xf32>
        %182 = vector.extract %175[0, 3] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %181 [%159, %168] : f32 into vector<2x8xf32>
        affine.yield %169, %183 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %142 : !frisk.async_task
      %subview_23 = memref.subview %arg2[0, %144, %110, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %147 = frisk.async_copy_gm_to_shm %subview_23 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = #map46} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %148 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %159 = affine.apply #map57(%arg11)[%thread_id_x]
        %160 = frisk.read_matrix_from_shm %view_21[%159] {element_bit_width = 16 : i64, frisk.v_double_buffer = 8192 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %161 = vector.bitcast %160 : vector<4xi32> to vector<8xf16>
        %162 = vector.extract %161[0] : f16 from vector<8xf16>
        %163 = vector.extract %161[%88] : f16 from vector<8xf16>
        %164 = vector.extract %161[1] : f16 from vector<8xf16>
        %165 = vector.extract %161[%89] : f16 from vector<8xf16>
        %166 = vector.extract %161[2] : f16 from vector<8xf16>
        %167 = vector.extract %161[%90] : f16 from vector<8xf16>
        %168 = vector.extract %161[3] : f16 from vector<8xf16>
        %169 = vector.extract %161[%91] : f16 from vector<8xf16>
        %170 = affine.apply #map52(%arg11)
        %171 = vector.insert %162, %arg12 [%170, 0] : f16 into vector<32x2xf16>
        %172 = vector.insert %163, %171 [%170, 1] : f16 into vector<32x2xf16>
        %173 = affine.apply #map53(%arg11)
        %174 = vector.insert %164, %172 [%173, 0] : f16 into vector<32x2xf16>
        %175 = vector.insert %165, %174 [%173, 1] : f16 into vector<32x2xf16>
        %176 = affine.apply #map54(%arg11)
        %177 = vector.insert %166, %175 [%176, 0] : f16 into vector<32x2xf16>
        %178 = vector.insert %167, %177 [%176, 1] : f16 into vector<32x2xf16>
        %179 = affine.apply #map55(%arg11)
        %180 = vector.insert %168, %178 [%179, 0] : f16 into vector<32x2xf16>
        %181 = vector.insert %169, %180 [%179, 1] : f16 into vector<32x2xf16>
        affine.yield %181 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_24 = memref.subview %arg2[0, %144, %110, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %149 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %146#1) -> (vector<2x8xf32>) {
        %159 = affine.apply #map4(%arg11)
        %160 = affine.apply #map3(%arg11)
        %161 = vector.extract %146#1[%159, %160] : f32 from vector<2x8xf32>
        %162 = affine.apply #map5(%arg11)
        %163 = vector.extract %146#1[%159, %162] : f32 from vector<2x8xf32>
        %164 = affine.apply #map6(%arg11)
        %165 = vector.extract %146#1[%159, %164] : f32 from vector<2x8xf32>
        %166 = affine.apply #map7(%arg11)
        %167 = vector.extract %146#1[%159, %166] : f32 from vector<2x8xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = vector.extract %139[%159, 0] : f16 from vector<2x32xf16>
        %170 = vector.extract %139[%159, 1] : f16 from vector<2x32xf16>
        %171 = vector.extract %139[%159, 2] : f16 from vector<2x32xf16>
        %172 = vector.extract %139[%159, 3] : f16 from vector<2x32xf16>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf16>
        %174 = affine.apply #map58(%arg11)
        %175 = vector.extract %148[0, %174] : f16 from vector<32x2xf16>
        %176 = vector.extract %148[1, %174] : f16 from vector<32x2xf16>
        %177 = vector.extract %148[2, %174] : f16 from vector<32x2xf16>
        %178 = vector.extract %148[3, %174] : f16 from vector<32x2xf16>
        %179 = vector.from_elements %175, %176, %177, %178 : vector<4x1xf16>
        %180 = frisk.warp_mma_rr %173, %179, %168 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %181 = vector.extract %139[%159, 4] : f16 from vector<2x32xf16>
        %182 = vector.extract %139[%159, 5] : f16 from vector<2x32xf16>
        %183 = vector.extract %139[%159, 6] : f16 from vector<2x32xf16>
        %184 = vector.extract %139[%159, 7] : f16 from vector<2x32xf16>
        %185 = vector.from_elements %181, %182, %183, %184 : vector<1x4xf16>
        %186 = vector.extract %148[4, %174] : f16 from vector<32x2xf16>
        %187 = vector.extract %148[5, %174] : f16 from vector<32x2xf16>
        %188 = vector.extract %148[6, %174] : f16 from vector<32x2xf16>
        %189 = vector.extract %148[7, %174] : f16 from vector<32x2xf16>
        %190 = vector.from_elements %186, %187, %188, %189 : vector<4x1xf16>
        %191 = frisk.warp_mma_rr %185, %190, %180 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %192 = vector.extract %139[%159, 8] : f16 from vector<2x32xf16>
        %193 = vector.extract %139[%159, 9] : f16 from vector<2x32xf16>
        %194 = vector.extract %139[%159, 10] : f16 from vector<2x32xf16>
        %195 = vector.extract %139[%159, 11] : f16 from vector<2x32xf16>
        %196 = vector.from_elements %192, %193, %194, %195 : vector<1x4xf16>
        %197 = vector.extract %148[8, %174] : f16 from vector<32x2xf16>
        %198 = vector.extract %148[9, %174] : f16 from vector<32x2xf16>
        %199 = vector.extract %148[10, %174] : f16 from vector<32x2xf16>
        %200 = vector.extract %148[11, %174] : f16 from vector<32x2xf16>
        %201 = vector.from_elements %197, %198, %199, %200 : vector<4x1xf16>
        %202 = frisk.warp_mma_rr %196, %201, %191 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %203 = vector.extract %139[%159, 12] : f16 from vector<2x32xf16>
        %204 = vector.extract %139[%159, 13] : f16 from vector<2x32xf16>
        %205 = vector.extract %139[%159, 14] : f16 from vector<2x32xf16>
        %206 = vector.extract %139[%159, 15] : f16 from vector<2x32xf16>
        %207 = vector.from_elements %203, %204, %205, %206 : vector<1x4xf16>
        %208 = vector.extract %148[12, %174] : f16 from vector<32x2xf16>
        %209 = vector.extract %148[13, %174] : f16 from vector<32x2xf16>
        %210 = vector.extract %148[14, %174] : f16 from vector<32x2xf16>
        %211 = vector.extract %148[15, %174] : f16 from vector<32x2xf16>
        %212 = vector.from_elements %208, %209, %210, %211 : vector<4x1xf16>
        %213 = frisk.warp_mma_rr %207, %212, %202 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %214 = vector.extract %139[%159, 16] : f16 from vector<2x32xf16>
        %215 = vector.extract %139[%159, 17] : f16 from vector<2x32xf16>
        %216 = vector.extract %139[%159, 18] : f16 from vector<2x32xf16>
        %217 = vector.extract %139[%159, 19] : f16 from vector<2x32xf16>
        %218 = vector.from_elements %214, %215, %216, %217 : vector<1x4xf16>
        %219 = vector.extract %148[16, %174] : f16 from vector<32x2xf16>
        %220 = vector.extract %148[17, %174] : f16 from vector<32x2xf16>
        %221 = vector.extract %148[18, %174] : f16 from vector<32x2xf16>
        %222 = vector.extract %148[19, %174] : f16 from vector<32x2xf16>
        %223 = vector.from_elements %219, %220, %221, %222 : vector<4x1xf16>
        %224 = frisk.warp_mma_rr %218, %223, %213 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %225 = vector.extract %139[%159, 20] : f16 from vector<2x32xf16>
        %226 = vector.extract %139[%159, 21] : f16 from vector<2x32xf16>
        %227 = vector.extract %139[%159, 22] : f16 from vector<2x32xf16>
        %228 = vector.extract %139[%159, 23] : f16 from vector<2x32xf16>
        %229 = vector.from_elements %225, %226, %227, %228 : vector<1x4xf16>
        %230 = vector.extract %148[20, %174] : f16 from vector<32x2xf16>
        %231 = vector.extract %148[21, %174] : f16 from vector<32x2xf16>
        %232 = vector.extract %148[22, %174] : f16 from vector<32x2xf16>
        %233 = vector.extract %148[23, %174] : f16 from vector<32x2xf16>
        %234 = vector.from_elements %230, %231, %232, %233 : vector<4x1xf16>
        %235 = frisk.warp_mma_rr %229, %234, %224 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %236 = vector.extract %139[%159, 24] : f16 from vector<2x32xf16>
        %237 = vector.extract %139[%159, 25] : f16 from vector<2x32xf16>
        %238 = vector.extract %139[%159, 26] : f16 from vector<2x32xf16>
        %239 = vector.extract %139[%159, 27] : f16 from vector<2x32xf16>
        %240 = vector.from_elements %236, %237, %238, %239 : vector<1x4xf16>
        %241 = vector.extract %148[24, %174] : f16 from vector<32x2xf16>
        %242 = vector.extract %148[25, %174] : f16 from vector<32x2xf16>
        %243 = vector.extract %148[26, %174] : f16 from vector<32x2xf16>
        %244 = vector.extract %148[27, %174] : f16 from vector<32x2xf16>
        %245 = vector.from_elements %241, %242, %243, %244 : vector<4x1xf16>
        %246 = frisk.warp_mma_rr %240, %245, %235 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %247 = vector.extract %139[%159, 28] : f16 from vector<2x32xf16>
        %248 = vector.extract %139[%159, 29] : f16 from vector<2x32xf16>
        %249 = vector.extract %139[%159, 30] : f16 from vector<2x32xf16>
        %250 = vector.extract %139[%159, 31] : f16 from vector<2x32xf16>
        %251 = vector.from_elements %247, %248, %249, %250 : vector<1x4xf16>
        %252 = vector.extract %148[28, %174] : f16 from vector<32x2xf16>
        %253 = vector.extract %148[29, %174] : f16 from vector<32x2xf16>
        %254 = vector.extract %148[30, %174] : f16 from vector<32x2xf16>
        %255 = vector.extract %148[31, %174] : f16 from vector<32x2xf16>
        %256 = vector.from_elements %252, %253, %254, %255 : vector<4x1xf16>
        %257 = frisk.warp_mma_rr %251, %256, %246 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %258 = vector.extract %257[0, 0] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %arg12 [%159, %160] : f32 into vector<2x8xf32>
        %260 = vector.extract %257[0, 1] : f32 from vector<1x4xf32>
        %261 = vector.insert %260, %259 [%159, %162] : f32 into vector<2x8xf32>
        %262 = vector.extract %257[0, 2] : f32 from vector<1x4xf32>
        %263 = vector.insert %262, %261 [%159, %164] : f32 into vector<2x8xf32>
        %264 = vector.extract %257[0, 3] : f32 from vector<1x4xf32>
        %265 = vector.insert %264, %263 [%159, %166] : f32 into vector<2x8xf32>
        affine.yield %265 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %150:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %159 = affine.apply #map4(%arg11)
        %160 = vector.extract %135#1[%159, 0] : f32 from vector<2x1xf32>
        %161 = vector.broadcast %160 : f32 to vector<1x4xf32>
        %162 = affine.apply #map3(%arg11)
        %163 = vector.insert %160, %arg12 [%159, %162] : f32 into vector<2x8xf32>
        %164 = affine.apply #map5(%arg11)
        %165 = vector.insert %160, %163 [%159, %164] : f32 into vector<2x8xf32>
        %166 = affine.apply #map6(%arg11)
        %167 = vector.insert %160, %165 [%159, %166] : f32 into vector<2x8xf32>
        %168 = affine.apply #map7(%arg11)
        %169 = vector.insert %160, %167 [%159, %168] : f32 into vector<2x8xf32>
        %170 = vector.extract %arg9[%159, %162] : f32 from vector<2x8xf32>
        %171 = vector.extract %arg9[%159, %164] : f32 from vector<2x8xf32>
        %172 = vector.extract %arg9[%159, %166] : f32 from vector<2x8xf32>
        %173 = vector.extract %arg9[%159, %168] : f32 from vector<2x8xf32>
        %174 = vector.from_elements %170, %171, %172, %173 : vector<1x4xf32>
        %175 = arith.mulf %174, %161 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %176 = vector.extract %175[0, 0] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %arg13 [%159, %162] : f32 into vector<2x8xf32>
        %178 = vector.extract %175[0, 1] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%159, %164] : f32 into vector<2x8xf32>
        %180 = vector.extract %175[0, 2] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%159, %166] : f32 into vector<2x8xf32>
        %182 = vector.extract %175[0, 3] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %181 [%159, %168] : f32 into vector<2x8xf32>
        affine.yield %169, %183 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %147 : !frisk.async_task
      %subview_25 = memref.subview %arg2[0, %144, %110, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %c8192_26 = arith.constant 8192 : index
      %view_27 = memref.view %alloc[%c8192_26][] : memref<16384xi8, 3> to memref<4096xf16, 3>
      %c8192_28 = arith.constant 8192 : index
      %view_29 = memref.view %alloc[%c8192_28][] : memref<16384xi8, 3> to memref<4096xf16, 3>
      %151 = frisk.async_copy_gm_to_shm %subview_25 to %view_27 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = #map46} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %152 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %159 = affine.apply #map57(%arg11)[%thread_id_x]
        %160 = frisk.read_matrix_from_shm %view[%159] {element_bit_width = 16 : i64, frisk.v_double_buffer = 8192 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %161 = vector.bitcast %160 : vector<4xi32> to vector<8xf16>
        %162 = vector.extract %161[0] : f16 from vector<8xf16>
        %163 = vector.extract %161[%88] : f16 from vector<8xf16>
        %164 = vector.extract %161[1] : f16 from vector<8xf16>
        %165 = vector.extract %161[%89] : f16 from vector<8xf16>
        %166 = vector.extract %161[2] : f16 from vector<8xf16>
        %167 = vector.extract %161[%90] : f16 from vector<8xf16>
        %168 = vector.extract %161[3] : f16 from vector<8xf16>
        %169 = vector.extract %161[%91] : f16 from vector<8xf16>
        %170 = affine.apply #map52(%arg11)
        %171 = vector.insert %162, %arg12 [%170, 0] : f16 into vector<32x2xf16>
        %172 = vector.insert %163, %171 [%170, 1] : f16 into vector<32x2xf16>
        %173 = affine.apply #map53(%arg11)
        %174 = vector.insert %164, %172 [%173, 0] : f16 into vector<32x2xf16>
        %175 = vector.insert %165, %174 [%173, 1] : f16 into vector<32x2xf16>
        %176 = affine.apply #map54(%arg11)
        %177 = vector.insert %166, %175 [%176, 0] : f16 into vector<32x2xf16>
        %178 = vector.insert %167, %177 [%176, 1] : f16 into vector<32x2xf16>
        %179 = affine.apply #map55(%arg11)
        %180 = vector.insert %168, %178 [%179, 0] : f16 into vector<32x2xf16>
        %181 = vector.insert %169, %180 [%179, 1] : f16 into vector<32x2xf16>
        affine.yield %181 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_30 = memref.subview %arg2[0, %144, %110, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %153 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %150#1) -> (vector<2x8xf32>) {
        %159 = affine.apply #map4(%arg11)
        %160 = affine.apply #map3(%arg11)
        %161 = vector.extract %150#1[%159, %160] : f32 from vector<2x8xf32>
        %162 = affine.apply #map5(%arg11)
        %163 = vector.extract %150#1[%159, %162] : f32 from vector<2x8xf32>
        %164 = affine.apply #map6(%arg11)
        %165 = vector.extract %150#1[%159, %164] : f32 from vector<2x8xf32>
        %166 = affine.apply #map7(%arg11)
        %167 = vector.extract %150#1[%159, %166] : f32 from vector<2x8xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = vector.extract %139[%159, 0] : f16 from vector<2x32xf16>
        %170 = vector.extract %139[%159, 1] : f16 from vector<2x32xf16>
        %171 = vector.extract %139[%159, 2] : f16 from vector<2x32xf16>
        %172 = vector.extract %139[%159, 3] : f16 from vector<2x32xf16>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf16>
        %174 = affine.apply #map58(%arg11)
        %175 = vector.extract %152[0, %174] : f16 from vector<32x2xf16>
        %176 = vector.extract %152[1, %174] : f16 from vector<32x2xf16>
        %177 = vector.extract %152[2, %174] : f16 from vector<32x2xf16>
        %178 = vector.extract %152[3, %174] : f16 from vector<32x2xf16>
        %179 = vector.from_elements %175, %176, %177, %178 : vector<4x1xf16>
        %180 = frisk.warp_mma_rr %173, %179, %168 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %181 = vector.extract %139[%159, 4] : f16 from vector<2x32xf16>
        %182 = vector.extract %139[%159, 5] : f16 from vector<2x32xf16>
        %183 = vector.extract %139[%159, 6] : f16 from vector<2x32xf16>
        %184 = vector.extract %139[%159, 7] : f16 from vector<2x32xf16>
        %185 = vector.from_elements %181, %182, %183, %184 : vector<1x4xf16>
        %186 = vector.extract %152[4, %174] : f16 from vector<32x2xf16>
        %187 = vector.extract %152[5, %174] : f16 from vector<32x2xf16>
        %188 = vector.extract %152[6, %174] : f16 from vector<32x2xf16>
        %189 = vector.extract %152[7, %174] : f16 from vector<32x2xf16>
        %190 = vector.from_elements %186, %187, %188, %189 : vector<4x1xf16>
        %191 = frisk.warp_mma_rr %185, %190, %180 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %192 = vector.extract %139[%159, 8] : f16 from vector<2x32xf16>
        %193 = vector.extract %139[%159, 9] : f16 from vector<2x32xf16>
        %194 = vector.extract %139[%159, 10] : f16 from vector<2x32xf16>
        %195 = vector.extract %139[%159, 11] : f16 from vector<2x32xf16>
        %196 = vector.from_elements %192, %193, %194, %195 : vector<1x4xf16>
        %197 = vector.extract %152[8, %174] : f16 from vector<32x2xf16>
        %198 = vector.extract %152[9, %174] : f16 from vector<32x2xf16>
        %199 = vector.extract %152[10, %174] : f16 from vector<32x2xf16>
        %200 = vector.extract %152[11, %174] : f16 from vector<32x2xf16>
        %201 = vector.from_elements %197, %198, %199, %200 : vector<4x1xf16>
        %202 = frisk.warp_mma_rr %196, %201, %191 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %203 = vector.extract %139[%159, 12] : f16 from vector<2x32xf16>
        %204 = vector.extract %139[%159, 13] : f16 from vector<2x32xf16>
        %205 = vector.extract %139[%159, 14] : f16 from vector<2x32xf16>
        %206 = vector.extract %139[%159, 15] : f16 from vector<2x32xf16>
        %207 = vector.from_elements %203, %204, %205, %206 : vector<1x4xf16>
        %208 = vector.extract %152[12, %174] : f16 from vector<32x2xf16>
        %209 = vector.extract %152[13, %174] : f16 from vector<32x2xf16>
        %210 = vector.extract %152[14, %174] : f16 from vector<32x2xf16>
        %211 = vector.extract %152[15, %174] : f16 from vector<32x2xf16>
        %212 = vector.from_elements %208, %209, %210, %211 : vector<4x1xf16>
        %213 = frisk.warp_mma_rr %207, %212, %202 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %214 = vector.extract %139[%159, 16] : f16 from vector<2x32xf16>
        %215 = vector.extract %139[%159, 17] : f16 from vector<2x32xf16>
        %216 = vector.extract %139[%159, 18] : f16 from vector<2x32xf16>
        %217 = vector.extract %139[%159, 19] : f16 from vector<2x32xf16>
        %218 = vector.from_elements %214, %215, %216, %217 : vector<1x4xf16>
        %219 = vector.extract %152[16, %174] : f16 from vector<32x2xf16>
        %220 = vector.extract %152[17, %174] : f16 from vector<32x2xf16>
        %221 = vector.extract %152[18, %174] : f16 from vector<32x2xf16>
        %222 = vector.extract %152[19, %174] : f16 from vector<32x2xf16>
        %223 = vector.from_elements %219, %220, %221, %222 : vector<4x1xf16>
        %224 = frisk.warp_mma_rr %218, %223, %213 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %225 = vector.extract %139[%159, 20] : f16 from vector<2x32xf16>
        %226 = vector.extract %139[%159, 21] : f16 from vector<2x32xf16>
        %227 = vector.extract %139[%159, 22] : f16 from vector<2x32xf16>
        %228 = vector.extract %139[%159, 23] : f16 from vector<2x32xf16>
        %229 = vector.from_elements %225, %226, %227, %228 : vector<1x4xf16>
        %230 = vector.extract %152[20, %174] : f16 from vector<32x2xf16>
        %231 = vector.extract %152[21, %174] : f16 from vector<32x2xf16>
        %232 = vector.extract %152[22, %174] : f16 from vector<32x2xf16>
        %233 = vector.extract %152[23, %174] : f16 from vector<32x2xf16>
        %234 = vector.from_elements %230, %231, %232, %233 : vector<4x1xf16>
        %235 = frisk.warp_mma_rr %229, %234, %224 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %236 = vector.extract %139[%159, 24] : f16 from vector<2x32xf16>
        %237 = vector.extract %139[%159, 25] : f16 from vector<2x32xf16>
        %238 = vector.extract %139[%159, 26] : f16 from vector<2x32xf16>
        %239 = vector.extract %139[%159, 27] : f16 from vector<2x32xf16>
        %240 = vector.from_elements %236, %237, %238, %239 : vector<1x4xf16>
        %241 = vector.extract %152[24, %174] : f16 from vector<32x2xf16>
        %242 = vector.extract %152[25, %174] : f16 from vector<32x2xf16>
        %243 = vector.extract %152[26, %174] : f16 from vector<32x2xf16>
        %244 = vector.extract %152[27, %174] : f16 from vector<32x2xf16>
        %245 = vector.from_elements %241, %242, %243, %244 : vector<4x1xf16>
        %246 = frisk.warp_mma_rr %240, %245, %235 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %247 = vector.extract %139[%159, 28] : f16 from vector<2x32xf16>
        %248 = vector.extract %139[%159, 29] : f16 from vector<2x32xf16>
        %249 = vector.extract %139[%159, 30] : f16 from vector<2x32xf16>
        %250 = vector.extract %139[%159, 31] : f16 from vector<2x32xf16>
        %251 = vector.from_elements %247, %248, %249, %250 : vector<1x4xf16>
        %252 = vector.extract %152[28, %174] : f16 from vector<32x2xf16>
        %253 = vector.extract %152[29, %174] : f16 from vector<32x2xf16>
        %254 = vector.extract %152[30, %174] : f16 from vector<32x2xf16>
        %255 = vector.extract %152[31, %174] : f16 from vector<32x2xf16>
        %256 = vector.from_elements %252, %253, %254, %255 : vector<4x1xf16>
        %257 = frisk.warp_mma_rr %251, %256, %246 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %258 = vector.extract %257[0, 0] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %arg12 [%159, %160] : f32 into vector<2x8xf32>
        %260 = vector.extract %257[0, 1] : f32 from vector<1x4xf32>
        %261 = vector.insert %260, %259 [%159, %162] : f32 into vector<2x8xf32>
        %262 = vector.extract %257[0, 2] : f32 from vector<1x4xf32>
        %263 = vector.insert %262, %261 [%159, %164] : f32 into vector<2x8xf32>
        %264 = vector.extract %257[0, 3] : f32 from vector<1x4xf32>
        %265 = vector.insert %264, %263 [%159, %166] : f32 into vector<2x8xf32>
        affine.yield %265 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %154:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %159 = affine.apply #map4(%arg11)
        %160 = vector.extract %135#1[%159, 0] : f32 from vector<2x1xf32>
        %161 = vector.broadcast %160 : f32 to vector<1x4xf32>
        %162 = affine.apply #map3(%arg11)
        %163 = vector.insert %160, %arg12 [%159, %162] : f32 into vector<2x8xf32>
        %164 = affine.apply #map5(%arg11)
        %165 = vector.insert %160, %163 [%159, %164] : f32 into vector<2x8xf32>
        %166 = affine.apply #map6(%arg11)
        %167 = vector.insert %160, %165 [%159, %166] : f32 into vector<2x8xf32>
        %168 = affine.apply #map7(%arg11)
        %169 = vector.insert %160, %167 [%159, %168] : f32 into vector<2x8xf32>
        %170 = vector.extract %arg10[%159, %162] : f32 from vector<2x8xf32>
        %171 = vector.extract %arg10[%159, %164] : f32 from vector<2x8xf32>
        %172 = vector.extract %arg10[%159, %166] : f32 from vector<2x8xf32>
        %173 = vector.extract %arg10[%159, %168] : f32 from vector<2x8xf32>
        %174 = vector.from_elements %170, %171, %172, %173 : vector<1x4xf32>
        %175 = arith.mulf %174, %161 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %176 = vector.extract %175[0, 0] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %arg13 [%159, %162] : f32 into vector<2x8xf32>
        %178 = vector.extract %175[0, 1] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%159, %164] : f32 into vector<2x8xf32>
        %180 = vector.extract %175[0, 2] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%159, %166] : f32 into vector<2x8xf32>
        %182 = vector.extract %175[0, 3] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %181 [%159, %168] : f32 into vector<2x8xf32>
        affine.yield %169, %183 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %151 : !frisk.async_task
      %155 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %159 = affine.apply #map57(%arg11)[%thread_id_x]
        %160 = frisk.read_matrix_from_shm %view_29[%159] {element_bit_width = 16 : i64, frisk.v_double_buffer = 8192 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %161 = vector.bitcast %160 : vector<4xi32> to vector<8xf16>
        %162 = vector.extract %161[0] : f16 from vector<8xf16>
        %163 = vector.extract %161[%88] : f16 from vector<8xf16>
        %164 = vector.extract %161[1] : f16 from vector<8xf16>
        %165 = vector.extract %161[%89] : f16 from vector<8xf16>
        %166 = vector.extract %161[2] : f16 from vector<8xf16>
        %167 = vector.extract %161[%90] : f16 from vector<8xf16>
        %168 = vector.extract %161[3] : f16 from vector<8xf16>
        %169 = vector.extract %161[%91] : f16 from vector<8xf16>
        %170 = affine.apply #map52(%arg11)
        %171 = vector.insert %162, %arg12 [%170, 0] : f16 into vector<32x2xf16>
        %172 = vector.insert %163, %171 [%170, 1] : f16 into vector<32x2xf16>
        %173 = affine.apply #map53(%arg11)
        %174 = vector.insert %164, %172 [%173, 0] : f16 into vector<32x2xf16>
        %175 = vector.insert %165, %174 [%173, 1] : f16 into vector<32x2xf16>
        %176 = affine.apply #map54(%arg11)
        %177 = vector.insert %166, %175 [%176, 0] : f16 into vector<32x2xf16>
        %178 = vector.insert %167, %177 [%176, 1] : f16 into vector<32x2xf16>
        %179 = affine.apply #map55(%arg11)
        %180 = vector.insert %168, %178 [%179, 0] : f16 into vector<32x2xf16>
        %181 = vector.insert %169, %180 [%179, 1] : f16 into vector<32x2xf16>
        affine.yield %181 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %156:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %154#1, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %159 = affine.apply #map4(%arg11)
        %160 = affine.apply #map3(%arg11)
        %161 = vector.extract %154#1[%159, %160] : f32 from vector<2x8xf32>
        %162 = affine.apply #map5(%arg11)
        %163 = vector.extract %154#1[%159, %162] : f32 from vector<2x8xf32>
        %164 = affine.apply #map6(%arg11)
        %165 = vector.extract %154#1[%159, %164] : f32 from vector<2x8xf32>
        %166 = affine.apply #map7(%arg11)
        %167 = vector.extract %154#1[%159, %166] : f32 from vector<2x8xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = vector.extract %139[%159, 0] : f16 from vector<2x32xf16>
        %170 = vector.extract %139[%159, 1] : f16 from vector<2x32xf16>
        %171 = vector.extract %139[%159, 2] : f16 from vector<2x32xf16>
        %172 = vector.extract %139[%159, 3] : f16 from vector<2x32xf16>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf16>
        %174 = affine.apply #map58(%arg11)
        %175 = vector.extract %155[0, %174] : f16 from vector<32x2xf16>
        %176 = vector.extract %155[1, %174] : f16 from vector<32x2xf16>
        %177 = vector.extract %155[2, %174] : f16 from vector<32x2xf16>
        %178 = vector.extract %155[3, %174] : f16 from vector<32x2xf16>
        %179 = vector.from_elements %175, %176, %177, %178 : vector<4x1xf16>
        %180 = frisk.warp_mma_rr %173, %179, %168 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %181 = vector.extract %139[%159, 4] : f16 from vector<2x32xf16>
        %182 = vector.extract %139[%159, 5] : f16 from vector<2x32xf16>
        %183 = vector.extract %139[%159, 6] : f16 from vector<2x32xf16>
        %184 = vector.extract %139[%159, 7] : f16 from vector<2x32xf16>
        %185 = vector.from_elements %181, %182, %183, %184 : vector<1x4xf16>
        %186 = vector.extract %155[4, %174] : f16 from vector<32x2xf16>
        %187 = vector.extract %155[5, %174] : f16 from vector<32x2xf16>
        %188 = vector.extract %155[6, %174] : f16 from vector<32x2xf16>
        %189 = vector.extract %155[7, %174] : f16 from vector<32x2xf16>
        %190 = vector.from_elements %186, %187, %188, %189 : vector<4x1xf16>
        %191 = frisk.warp_mma_rr %185, %190, %180 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %192 = vector.extract %139[%159, 8] : f16 from vector<2x32xf16>
        %193 = vector.extract %139[%159, 9] : f16 from vector<2x32xf16>
        %194 = vector.extract %139[%159, 10] : f16 from vector<2x32xf16>
        %195 = vector.extract %139[%159, 11] : f16 from vector<2x32xf16>
        %196 = vector.from_elements %192, %193, %194, %195 : vector<1x4xf16>
        %197 = vector.extract %155[8, %174] : f16 from vector<32x2xf16>
        %198 = vector.extract %155[9, %174] : f16 from vector<32x2xf16>
        %199 = vector.extract %155[10, %174] : f16 from vector<32x2xf16>
        %200 = vector.extract %155[11, %174] : f16 from vector<32x2xf16>
        %201 = vector.from_elements %197, %198, %199, %200 : vector<4x1xf16>
        %202 = frisk.warp_mma_rr %196, %201, %191 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %203 = vector.extract %139[%159, 12] : f16 from vector<2x32xf16>
        %204 = vector.extract %139[%159, 13] : f16 from vector<2x32xf16>
        %205 = vector.extract %139[%159, 14] : f16 from vector<2x32xf16>
        %206 = vector.extract %139[%159, 15] : f16 from vector<2x32xf16>
        %207 = vector.from_elements %203, %204, %205, %206 : vector<1x4xf16>
        %208 = vector.extract %155[12, %174] : f16 from vector<32x2xf16>
        %209 = vector.extract %155[13, %174] : f16 from vector<32x2xf16>
        %210 = vector.extract %155[14, %174] : f16 from vector<32x2xf16>
        %211 = vector.extract %155[15, %174] : f16 from vector<32x2xf16>
        %212 = vector.from_elements %208, %209, %210, %211 : vector<4x1xf16>
        %213 = frisk.warp_mma_rr %207, %212, %202 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %214 = vector.extract %139[%159, 16] : f16 from vector<2x32xf16>
        %215 = vector.extract %139[%159, 17] : f16 from vector<2x32xf16>
        %216 = vector.extract %139[%159, 18] : f16 from vector<2x32xf16>
        %217 = vector.extract %139[%159, 19] : f16 from vector<2x32xf16>
        %218 = vector.from_elements %214, %215, %216, %217 : vector<1x4xf16>
        %219 = vector.extract %155[16, %174] : f16 from vector<32x2xf16>
        %220 = vector.extract %155[17, %174] : f16 from vector<32x2xf16>
        %221 = vector.extract %155[18, %174] : f16 from vector<32x2xf16>
        %222 = vector.extract %155[19, %174] : f16 from vector<32x2xf16>
        %223 = vector.from_elements %219, %220, %221, %222 : vector<4x1xf16>
        %224 = frisk.warp_mma_rr %218, %223, %213 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %225 = vector.extract %139[%159, 20] : f16 from vector<2x32xf16>
        %226 = vector.extract %139[%159, 21] : f16 from vector<2x32xf16>
        %227 = vector.extract %139[%159, 22] : f16 from vector<2x32xf16>
        %228 = vector.extract %139[%159, 23] : f16 from vector<2x32xf16>
        %229 = vector.from_elements %225, %226, %227, %228 : vector<1x4xf16>
        %230 = vector.extract %155[20, %174] : f16 from vector<32x2xf16>
        %231 = vector.extract %155[21, %174] : f16 from vector<32x2xf16>
        %232 = vector.extract %155[22, %174] : f16 from vector<32x2xf16>
        %233 = vector.extract %155[23, %174] : f16 from vector<32x2xf16>
        %234 = vector.from_elements %230, %231, %232, %233 : vector<4x1xf16>
        %235 = frisk.warp_mma_rr %229, %234, %224 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %236 = vector.extract %139[%159, 24] : f16 from vector<2x32xf16>
        %237 = vector.extract %139[%159, 25] : f16 from vector<2x32xf16>
        %238 = vector.extract %139[%159, 26] : f16 from vector<2x32xf16>
        %239 = vector.extract %139[%159, 27] : f16 from vector<2x32xf16>
        %240 = vector.from_elements %236, %237, %238, %239 : vector<1x4xf16>
        %241 = vector.extract %155[24, %174] : f16 from vector<32x2xf16>
        %242 = vector.extract %155[25, %174] : f16 from vector<32x2xf16>
        %243 = vector.extract %155[26, %174] : f16 from vector<32x2xf16>
        %244 = vector.extract %155[27, %174] : f16 from vector<32x2xf16>
        %245 = vector.from_elements %241, %242, %243, %244 : vector<4x1xf16>
        %246 = frisk.warp_mma_rr %240, %245, %235 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %247 = vector.extract %139[%159, 28] : f16 from vector<2x32xf16>
        %248 = vector.extract %139[%159, 29] : f16 from vector<2x32xf16>
        %249 = vector.extract %139[%159, 30] : f16 from vector<2x32xf16>
        %250 = vector.extract %139[%159, 31] : f16 from vector<2x32xf16>
        %251 = vector.from_elements %247, %248, %249, %250 : vector<1x4xf16>
        %252 = vector.extract %155[28, %174] : f16 from vector<32x2xf16>
        %253 = vector.extract %155[29, %174] : f16 from vector<32x2xf16>
        %254 = vector.extract %155[30, %174] : f16 from vector<32x2xf16>
        %255 = vector.extract %155[31, %174] : f16 from vector<32x2xf16>
        %256 = vector.from_elements %252, %253, %254, %255 : vector<4x1xf16>
        %257 = frisk.warp_mma_rr %251, %256, %246 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %258 = vector.extract %257[0, 0] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %arg12 [%159, %160] : f32 into vector<2x8xf32>
        %260 = vector.extract %257[0, 1] : f32 from vector<1x4xf32>
        %261 = vector.insert %260, %259 [%159, %162] : f32 into vector<2x8xf32>
        %262 = vector.extract %257[0, 2] : f32 from vector<1x4xf32>
        %263 = vector.insert %262, %261 [%159, %164] : f32 into vector<2x8xf32>
        %264 = vector.extract %257[0, 3] : f32 from vector<1x4xf32>
        %265 = vector.insert %264, %263 [%159, %166] : f32 into vector<2x8xf32>
        %266 = vector.extract %145[%159, %160] : f32 from vector<2x8xf32>
        %267 = vector.extract %145[%159, %162] : f32 from vector<2x8xf32>
        %268 = vector.extract %145[%159, %164] : f32 from vector<2x8xf32>
        %269 = vector.extract %145[%159, %166] : f32 from vector<2x8xf32>
        %270 = vector.insert %266, %arg13 [%159, %160] : f32 into vector<2x8xf32>
        %271 = vector.insert %267, %270 [%159, %162] : f32 into vector<2x8xf32>
        %272 = vector.insert %268, %271 [%159, %164] : f32 into vector<2x8xf32>
        %273 = vector.insert %269, %272 [%159, %166] : f32 into vector<2x8xf32>
        affine.yield %265, %273 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %157:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %159 = vector.extract %138[%arg11, 0] : f32 from vector<2x1xf32>
        %160 = vector.insert %159, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %161 = vector.extract %133[%arg11, 0] : f32 from vector<2x1xf32>
        %162 = vector.insert %161, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %160, %162 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %158:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5, %arg14 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %159 = affine.apply #map4(%arg11)
        %160 = affine.apply #map3(%arg11)
        %161 = vector.extract %149[%159, %160] : f32 from vector<2x8xf32>
        %162 = affine.apply #map5(%arg11)
        %163 = vector.extract %149[%159, %162] : f32 from vector<2x8xf32>
        %164 = affine.apply #map6(%arg11)
        %165 = vector.extract %149[%159, %164] : f32 from vector<2x8xf32>
        %166 = affine.apply #map7(%arg11)
        %167 = vector.extract %149[%159, %166] : f32 from vector<2x8xf32>
        %168 = vector.insert %161, %arg12 [%159, %160] : f32 into vector<2x8xf32>
        %169 = vector.insert %163, %168 [%159, %162] : f32 into vector<2x8xf32>
        %170 = vector.insert %165, %169 [%159, %164] : f32 into vector<2x8xf32>
        %171 = vector.insert %167, %170 [%159, %166] : f32 into vector<2x8xf32>
        %172 = vector.extract %153[%159, %160] : f32 from vector<2x8xf32>
        %173 = vector.extract %153[%159, %162] : f32 from vector<2x8xf32>
        %174 = vector.extract %153[%159, %164] : f32 from vector<2x8xf32>
        %175 = vector.extract %153[%159, %166] : f32 from vector<2x8xf32>
        %176 = vector.insert %172, %arg13 [%159, %160] : f32 into vector<2x8xf32>
        %177 = vector.insert %173, %176 [%159, %162] : f32 into vector<2x8xf32>
        %178 = vector.insert %174, %177 [%159, %164] : f32 into vector<2x8xf32>
        %179 = vector.insert %175, %178 [%159, %166] : f32 into vector<2x8xf32>
        %180 = vector.extract %156#0[%159, %160] : f32 from vector<2x8xf32>
        %181 = vector.extract %156#0[%159, %162] : f32 from vector<2x8xf32>
        %182 = vector.extract %156#0[%159, %164] : f32 from vector<2x8xf32>
        %183 = vector.extract %156#0[%159, %166] : f32 from vector<2x8xf32>
        %184 = vector.insert %180, %arg14 [%159, %160] : f32 into vector<2x8xf32>
        %185 = vector.insert %181, %184 [%159, %162] : f32 into vector<2x8xf32>
        %186 = vector.insert %182, %185 [%159, %164] : f32 into vector<2x8xf32>
        %187 = vector.insert %183, %186 [%159, %166] : f32 into vector<2x8xf32>
        affine.yield %171, %179, %187 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      gpu.barrier
      affine.yield %156#1, %157#0, %157#1, %158#0, %158#1, %158#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %93 = arith.index_cast %thread_id_x : index to i32
    %94 = arith.andi %93, %c16_i32 : i32
    %95 = arith.cmpi ne, %94, %c0_i32 : i32
    %96 = arith.andi %93, %c32_i32 : i32
    %97 = arith.cmpi ne, %96, %c0_i32 : i32
    %98 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %108 = affine.apply #map3(%arg4)
      %109 = affine.apply #map4(%arg4)
      %110 = vector.extract %92#5[%109, %108] : f32 from vector<2x8xf32>
      %111 = affine.apply #map5(%arg4)
      %112 = vector.extract %92#5[%109, %111] : f32 from vector<2x8xf32>
      %113 = affine.apply #map6(%arg4)
      %114 = vector.extract %92#5[%109, %113] : f32 from vector<2x8xf32>
      %115 = affine.apply #map7(%arg4)
      %116 = vector.extract %92#5[%109, %115] : f32 from vector<2x8xf32>
      %117 = arith.select %95, %110, %112 : f32
      %shuffleResult, %valid = gpu.shuffle xor %117, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %95, %shuffleResult, %110 : f32
      %119 = arith.select %95, %112, %shuffleResult : f32
      %120 = arith.select %95, %114, %116 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %120, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %95, %shuffleResult_12, %114 : f32
      %122 = arith.select %95, %116, %shuffleResult_12 : f32
      %123 = arith.select %97, %118, %121 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %123, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %124 = arith.select %97, %shuffleResult_14, %118 : f32
      %125 = arith.select %97, %121, %shuffleResult_14 : f32
      %126 = arith.select %97, %119, %122 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %126, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %127 = arith.select %97, %shuffleResult_16, %119 : f32
      %128 = arith.select %97, %122, %shuffleResult_16 : f32
      %129 = vector.insert %124, %arg5 [%109, %108] : f32 into vector<2x8xf32>
      %130 = vector.insert %127, %129 [%109, %111] : f32 into vector<2x8xf32>
      %131 = vector.insert %125, %130 [%109, %113] : f32 into vector<2x8xf32>
      %132 = vector.insert %128, %131 [%109, %115] : f32 into vector<2x8xf32>
      affine.yield %132 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %99 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %108 = affine.apply #map3(%arg4)
      %109 = affine.apply #map4(%arg4)
      %110 = vector.extract %92#4[%109, %108] : f32 from vector<2x8xf32>
      %111 = affine.apply #map5(%arg4)
      %112 = vector.extract %92#4[%109, %111] : f32 from vector<2x8xf32>
      %113 = affine.apply #map6(%arg4)
      %114 = vector.extract %92#4[%109, %113] : f32 from vector<2x8xf32>
      %115 = affine.apply #map7(%arg4)
      %116 = vector.extract %92#4[%109, %115] : f32 from vector<2x8xf32>
      %117 = arith.select %95, %110, %112 : f32
      %shuffleResult, %valid = gpu.shuffle xor %117, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %95, %shuffleResult, %110 : f32
      %119 = arith.select %95, %112, %shuffleResult : f32
      %120 = arith.select %95, %114, %116 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %120, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %95, %shuffleResult_12, %114 : f32
      %122 = arith.select %95, %116, %shuffleResult_12 : f32
      %123 = arith.select %97, %118, %121 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %123, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %124 = arith.select %97, %shuffleResult_14, %118 : f32
      %125 = arith.select %97, %121, %shuffleResult_14 : f32
      %126 = arith.select %97, %119, %122 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %126, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %127 = arith.select %97, %shuffleResult_16, %119 : f32
      %128 = arith.select %97, %122, %shuffleResult_16 : f32
      %129 = vector.insert %124, %arg5 [%109, %108] : f32 into vector<2x8xf32>
      %130 = vector.insert %127, %129 [%109, %111] : f32 into vector<2x8xf32>
      %131 = vector.insert %125, %130 [%109, %113] : f32 into vector<2x8xf32>
      %132 = vector.insert %128, %131 [%109, %115] : f32 into vector<2x8xf32>
      affine.yield %132 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %100 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %108 = affine.apply #map3(%arg4)
      %109 = affine.apply #map4(%arg4)
      %110 = vector.extract %92#3[%109, %108] : f32 from vector<2x8xf32>
      %111 = affine.apply #map5(%arg4)
      %112 = vector.extract %92#3[%109, %111] : f32 from vector<2x8xf32>
      %113 = affine.apply #map6(%arg4)
      %114 = vector.extract %92#3[%109, %113] : f32 from vector<2x8xf32>
      %115 = affine.apply #map7(%arg4)
      %116 = vector.extract %92#3[%109, %115] : f32 from vector<2x8xf32>
      %117 = arith.select %95, %110, %112 : f32
      %shuffleResult, %valid = gpu.shuffle xor %117, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %95, %shuffleResult, %110 : f32
      %119 = arith.select %95, %112, %shuffleResult : f32
      %120 = arith.select %95, %114, %116 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %120, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %95, %shuffleResult_12, %114 : f32
      %122 = arith.select %95, %116, %shuffleResult_12 : f32
      %123 = arith.select %97, %118, %121 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %123, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %124 = arith.select %97, %shuffleResult_14, %118 : f32
      %125 = arith.select %97, %121, %shuffleResult_14 : f32
      %126 = arith.select %97, %119, %122 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %126, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %127 = arith.select %97, %shuffleResult_16, %119 : f32
      %128 = arith.select %97, %122, %shuffleResult_16 : f32
      %129 = vector.insert %124, %arg5 [%109, %108] : f32 into vector<2x8xf32>
      %130 = vector.insert %127, %129 [%109, %111] : f32 into vector<2x8xf32>
      %131 = vector.insert %125, %130 [%109, %113] : f32 into vector<2x8xf32>
      %132 = vector.insert %128, %131 [%109, %115] : f32 into vector<2x8xf32>
      affine.yield %132 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %101 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %108 = affine.apply #map3(%arg4)
      %109 = affine.apply #map4(%arg4)
      %110 = vector.extract %92#0[%109, %108] : f32 from vector<2x8xf32>
      %111 = affine.apply #map5(%arg4)
      %112 = vector.extract %92#0[%109, %111] : f32 from vector<2x8xf32>
      %113 = affine.apply #map6(%arg4)
      %114 = vector.extract %92#0[%109, %113] : f32 from vector<2x8xf32>
      %115 = affine.apply #map7(%arg4)
      %116 = vector.extract %92#0[%109, %115] : f32 from vector<2x8xf32>
      %117 = arith.select %95, %110, %112 : f32
      %shuffleResult, %valid = gpu.shuffle xor %117, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %95, %shuffleResult, %110 : f32
      %119 = arith.select %95, %112, %shuffleResult : f32
      %120 = arith.select %95, %114, %116 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %120, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %95, %shuffleResult_12, %114 : f32
      %122 = arith.select %95, %116, %shuffleResult_12 : f32
      %123 = arith.select %97, %118, %121 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %123, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %124 = arith.select %97, %shuffleResult_14, %118 : f32
      %125 = arith.select %97, %121, %shuffleResult_14 : f32
      %126 = arith.select %97, %119, %122 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %126, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %127 = arith.select %97, %shuffleResult_16, %119 : f32
      %128 = arith.select %97, %122, %shuffleResult_16 : f32
      %129 = vector.insert %124, %arg5 [%109, %108] : f32 into vector<2x8xf32>
      %130 = vector.insert %127, %129 [%109, %111] : f32 into vector<2x8xf32>
      %131 = vector.insert %125, %130 [%109, %113] : f32 into vector<2x8xf32>
      %132 = vector.insert %128, %131 [%109, %115] : f32 into vector<2x8xf32>
      affine.yield %132 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %102:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %108 = affine.apply #map4(%arg4)
      %109 = vector.extract %92#1[%108, 0] : f32 from vector<2x1xf32>
      %110 = vector.broadcast %109 : f32 to vector<1x4xf32>
      %111 = affine.apply #map3(%arg4)
      %112 = vector.insert %109, %arg5 [%108, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply #map5(%arg4)
      %114 = vector.insert %109, %112 [%108, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply #map6(%arg4)
      %116 = vector.insert %109, %114 [%108, %115] : f32 into vector<2x8xf32>
      %117 = affine.apply #map7(%arg4)
      %118 = vector.insert %109, %116 [%108, %117] : f32 into vector<2x8xf32>
      %119 = vector.extract %101[%108, %111] : f32 from vector<2x8xf32>
      %120 = vector.extract %101[%108, %113] : f32 from vector<2x8xf32>
      %121 = vector.extract %101[%108, %115] : f32 from vector<2x8xf32>
      %122 = vector.extract %101[%108, %117] : f32 from vector<2x8xf32>
      %123 = vector.from_elements %119, %120, %121, %122 : vector<1x4xf32>
      %124 = arith.divf %123, %110 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %125 = vector.extract %124[0, 0] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %arg6 [%108, %111] : f32 into vector<2x8xf32>
      %127 = vector.extract %124[0, 1] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%108, %113] : f32 into vector<2x8xf32>
      %129 = vector.extract %124[0, 2] : f32 from vector<1x4xf32>
      %130 = vector.insert %129, %128 [%108, %115] : f32 into vector<2x8xf32>
      %131 = vector.extract %124[0, 3] : f32 from vector<1x4xf32>
      %132 = vector.insert %131, %130 [%108, %117] : f32 into vector<2x8xf32>
      %133 = arith.truncf %124 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %134 = vector.extract %133[0, 0] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %arg7 [%108, %111] : f16 into vector<2x8xf16>
      %136 = vector.extract %133[0, 1] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%108, %113] : f16 into vector<2x8xf16>
      %138 = vector.extract %133[0, 2] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%108, %115] : f16 into vector<2x8xf16>
      %140 = vector.extract %133[0, 3] : f16 from vector<1x4xf16>
      %141 = vector.insert %140, %139 [%108, %117] : f16 into vector<2x8xf16>
      %142 = vector.insert %134, %arg8 [%108, %111] : f16 into vector<2x8xf16>
      %143 = vector.insert %136, %142 [%108, %113] : f16 into vector<2x8xf16>
      %144 = vector.insert %138, %143 [%108, %115] : f16 into vector<2x8xf16>
      %145 = vector.insert %140, %144 [%108, %117] : f16 into vector<2x8xf16>
      affine.yield %118, %132, %141, %145 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %103 = affine.apply #map59()[%thread_id_x]
    %104 = arith.cmpi ult, %103, %c4 : index
    scf.if %104 {
      affine.for %arg4 = 0 to 4 {
        %108 = affine.apply #map4(%arg4)
        %109 = affine.apply #map3(%arg4)
        %110 = vector.extract %102#3[%108, %109] : f16 from vector<2x8xf16>
        %111 = affine.apply #map5(%arg4)
        %112 = vector.extract %102#3[%108, %111] : f16 from vector<2x8xf16>
        %113 = affine.apply #map6(%arg4)
        %114 = vector.extract %102#3[%108, %113] : f16 from vector<2x8xf16>
        %115 = affine.apply #map7(%arg4)
        %116 = vector.extract %102#3[%108, %115] : f16 from vector<2x8xf16>
        affine.store %110, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %112, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %114, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %116, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %105:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %108 = affine.apply #map4(%arg4)
      %109 = vector.extract %92#1[%108, 0] : f32 from vector<2x1xf32>
      %110 = vector.broadcast %109 : f32 to vector<1x4xf32>
      %111 = affine.apply #map3(%arg4)
      %112 = vector.insert %109, %arg5 [%108, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply #map5(%arg4)
      %114 = vector.insert %109, %112 [%108, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply #map6(%arg4)
      %116 = vector.insert %109, %114 [%108, %115] : f32 into vector<2x8xf32>
      %117 = affine.apply #map7(%arg4)
      %118 = vector.insert %109, %116 [%108, %117] : f32 into vector<2x8xf32>
      %119 = vector.extract %100[%108, %111] : f32 from vector<2x8xf32>
      %120 = vector.extract %100[%108, %113] : f32 from vector<2x8xf32>
      %121 = vector.extract %100[%108, %115] : f32 from vector<2x8xf32>
      %122 = vector.extract %100[%108, %117] : f32 from vector<2x8xf32>
      %123 = vector.from_elements %119, %120, %121, %122 : vector<1x4xf32>
      %124 = arith.divf %123, %110 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %125 = vector.extract %124[0, 0] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %arg6 [%108, %111] : f32 into vector<2x8xf32>
      %127 = vector.extract %124[0, 1] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%108, %113] : f32 into vector<2x8xf32>
      %129 = vector.extract %124[0, 2] : f32 from vector<1x4xf32>
      %130 = vector.insert %129, %128 [%108, %115] : f32 into vector<2x8xf32>
      %131 = vector.extract %124[0, 3] : f32 from vector<1x4xf32>
      %132 = vector.insert %131, %130 [%108, %117] : f32 into vector<2x8xf32>
      %133 = arith.truncf %124 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %134 = vector.extract %133[0, 0] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %arg7 [%108, %111] : f16 into vector<2x8xf16>
      %136 = vector.extract %133[0, 1] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%108, %113] : f16 into vector<2x8xf16>
      %138 = vector.extract %133[0, 2] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%108, %115] : f16 into vector<2x8xf16>
      %140 = vector.extract %133[0, 3] : f16 from vector<1x4xf16>
      %141 = vector.insert %140, %139 [%108, %117] : f16 into vector<2x8xf16>
      %142 = vector.insert %134, %arg8 [%108, %111] : f16 into vector<2x8xf16>
      %143 = vector.insert %136, %142 [%108, %113] : f16 into vector<2x8xf16>
      %144 = vector.insert %138, %143 [%108, %115] : f16 into vector<2x8xf16>
      %145 = vector.insert %140, %144 [%108, %117] : f16 into vector<2x8xf16>
      affine.yield %118, %132, %141, %145 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %104 {
      affine.for %arg4 = 0 to 4 {
        %108 = affine.apply #map4(%arg4)
        %109 = affine.apply #map3(%arg4)
        %110 = vector.extract %105#3[%108, %109] : f16 from vector<2x8xf16>
        %111 = affine.apply #map5(%arg4)
        %112 = vector.extract %105#3[%108, %111] : f16 from vector<2x8xf16>
        %113 = affine.apply #map6(%arg4)
        %114 = vector.extract %105#3[%108, %113] : f16 from vector<2x8xf16>
        %115 = affine.apply #map7(%arg4)
        %116 = vector.extract %105#3[%108, %115] : f16 from vector<2x8xf16>
        affine.store %110, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %112, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %114, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %116, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %106:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %108 = affine.apply #map4(%arg4)
      %109 = vector.extract %92#1[%108, 0] : f32 from vector<2x1xf32>
      %110 = vector.broadcast %109 : f32 to vector<1x4xf32>
      %111 = affine.apply #map3(%arg4)
      %112 = vector.insert %109, %arg5 [%108, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply #map5(%arg4)
      %114 = vector.insert %109, %112 [%108, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply #map6(%arg4)
      %116 = vector.insert %109, %114 [%108, %115] : f32 into vector<2x8xf32>
      %117 = affine.apply #map7(%arg4)
      %118 = vector.insert %109, %116 [%108, %117] : f32 into vector<2x8xf32>
      %119 = vector.extract %99[%108, %111] : f32 from vector<2x8xf32>
      %120 = vector.extract %99[%108, %113] : f32 from vector<2x8xf32>
      %121 = vector.extract %99[%108, %115] : f32 from vector<2x8xf32>
      %122 = vector.extract %99[%108, %117] : f32 from vector<2x8xf32>
      %123 = vector.from_elements %119, %120, %121, %122 : vector<1x4xf32>
      %124 = arith.divf %123, %110 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %125 = vector.extract %124[0, 0] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %arg6 [%108, %111] : f32 into vector<2x8xf32>
      %127 = vector.extract %124[0, 1] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%108, %113] : f32 into vector<2x8xf32>
      %129 = vector.extract %124[0, 2] : f32 from vector<1x4xf32>
      %130 = vector.insert %129, %128 [%108, %115] : f32 into vector<2x8xf32>
      %131 = vector.extract %124[0, 3] : f32 from vector<1x4xf32>
      %132 = vector.insert %131, %130 [%108, %117] : f32 into vector<2x8xf32>
      %133 = arith.truncf %124 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %134 = vector.extract %133[0, 0] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %arg7 [%108, %111] : f16 into vector<2x8xf16>
      %136 = vector.extract %133[0, 1] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%108, %113] : f16 into vector<2x8xf16>
      %138 = vector.extract %133[0, 2] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%108, %115] : f16 into vector<2x8xf16>
      %140 = vector.extract %133[0, 3] : f16 from vector<1x4xf16>
      %141 = vector.insert %140, %139 [%108, %117] : f16 into vector<2x8xf16>
      %142 = vector.insert %134, %arg8 [%108, %111] : f16 into vector<2x8xf16>
      %143 = vector.insert %136, %142 [%108, %113] : f16 into vector<2x8xf16>
      %144 = vector.insert %138, %143 [%108, %115] : f16 into vector<2x8xf16>
      %145 = vector.insert %140, %144 [%108, %117] : f16 into vector<2x8xf16>
      affine.yield %118, %132, %141, %145 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %104 {
      affine.for %arg4 = 0 to 4 {
        %108 = affine.apply #map4(%arg4)
        %109 = affine.apply #map3(%arg4)
        %110 = vector.extract %106#3[%108, %109] : f16 from vector<2x8xf16>
        %111 = affine.apply #map5(%arg4)
        %112 = vector.extract %106#3[%108, %111] : f16 from vector<2x8xf16>
        %113 = affine.apply #map6(%arg4)
        %114 = vector.extract %106#3[%108, %113] : f16 from vector<2x8xf16>
        %115 = affine.apply #map7(%arg4)
        %116 = vector.extract %106#3[%108, %115] : f16 from vector<2x8xf16>
        affine.store %110, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %112, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %114, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %116, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %107:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %108 = affine.apply #map4(%arg4)
      %109 = vector.extract %92#1[%108, 0] : f32 from vector<2x1xf32>
      %110 = vector.broadcast %109 : f32 to vector<1x4xf32>
      %111 = affine.apply #map3(%arg4)
      %112 = vector.insert %109, %arg5 [%108, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply #map5(%arg4)
      %114 = vector.insert %109, %112 [%108, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply #map6(%arg4)
      %116 = vector.insert %109, %114 [%108, %115] : f32 into vector<2x8xf32>
      %117 = affine.apply #map7(%arg4)
      %118 = vector.insert %109, %116 [%108, %117] : f32 into vector<2x8xf32>
      %119 = vector.extract %98[%108, %111] : f32 from vector<2x8xf32>
      %120 = vector.extract %98[%108, %113] : f32 from vector<2x8xf32>
      %121 = vector.extract %98[%108, %115] : f32 from vector<2x8xf32>
      %122 = vector.extract %98[%108, %117] : f32 from vector<2x8xf32>
      %123 = vector.from_elements %119, %120, %121, %122 : vector<1x4xf32>
      %124 = arith.divf %123, %110 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %125 = vector.extract %124[0, 0] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %arg6 [%108, %111] : f32 into vector<2x8xf32>
      %127 = vector.extract %124[0, 1] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%108, %113] : f32 into vector<2x8xf32>
      %129 = vector.extract %124[0, 2] : f32 from vector<1x4xf32>
      %130 = vector.insert %129, %128 [%108, %115] : f32 into vector<2x8xf32>
      %131 = vector.extract %124[0, 3] : f32 from vector<1x4xf32>
      %132 = vector.insert %131, %130 [%108, %117] : f32 into vector<2x8xf32>
      %133 = arith.truncf %124 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %134 = vector.extract %133[0, 0] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %arg7 [%108, %111] : f16 into vector<2x8xf16>
      %136 = vector.extract %133[0, 1] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%108, %113] : f16 into vector<2x8xf16>
      %138 = vector.extract %133[0, 2] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%108, %115] : f16 into vector<2x8xf16>
      %140 = vector.extract %133[0, 3] : f16 from vector<1x4xf16>
      %141 = vector.insert %140, %139 [%108, %117] : f16 into vector<2x8xf16>
      %142 = vector.insert %134, %arg8 [%108, %111] : f16 into vector<2x8xf16>
      %143 = vector.insert %136, %142 [%108, %113] : f16 into vector<2x8xf16>
      %144 = vector.insert %138, %143 [%108, %115] : f16 into vector<2x8xf16>
      %145 = vector.insert %140, %144 [%108, %117] : f16 into vector<2x8xf16>
      affine.yield %118, %132, %141, %145 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %104 {
      affine.for %arg4 = 0 to 4 {
        %108 = affine.apply #map4(%arg4)
        %109 = affine.apply #map3(%arg4)
        %110 = vector.extract %107#3[%108, %109] : f16 from vector<2x8xf16>
        %111 = affine.apply #map5(%arg4)
        %112 = vector.extract %107#3[%108, %111] : f16 from vector<2x8xf16>
        %113 = affine.apply #map6(%arg4)
        %114 = vector.extract %107#3[%108, %113] : f16 from vector<2x8xf16>
        %115 = affine.apply #map7(%arg4)
        %116 = vector.extract %107#3[%108, %115] : f16 from vector<2x8xf16>
        affine.store %110, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %112, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %114, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %116, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}

