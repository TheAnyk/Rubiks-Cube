import random
import numpy as np
import itertools
import copy
import warnings

import consts


class Cube:
    def __init__(self, *save_strategy, faces_factory=None, cubies_factory=None, turns_factory=None):
        """
        Initializes Cube object
        Args:
            save_strategy (sequence[str]): How should the cube be saved, can be multiple at once. Use "faces" to store in a human-readable format, "cubies" to store in a compressed format, and "turns" to store in turns - this strategy cannot be visualized. Defaults to "faces"

        Keyword args:
            faces_factory (callable): Function used to create default faces array. It has to return a 2D numpy array like object of shape (6, 8)
            cubies_factory (callable): Function used to create default cubies array. It has to return a 2D numpy array like object of shape (20, 2)
            turns_factory (callable): Function used to create default turns array. It has to return a object that supports ".append()" and indexing
        """
        if not (isinstance(save_strategy, tuple) and all(isinstance(it, str) for it in save_strategy)):
            raise ValueError("save_strategy has to be a sequence of strings")
        
        if "faces" in save_strategy:
            self.faces_factory = faces_factory or (lambda: np.arange(6).view(6, 1).repeat(8, axis=1))
        if "cubies" in save_strategy:
            raise NotImplementedError("Cubies turns are not implemented yet")
            self.cubies_factory = cubies_factory or (lambda: np.arange(20*2).view(20, 2))
        if "turns" in save_strategy:
            self.turns_factory = turns_factory or (lambda: [])

        self.save_strategy = save_strategy

        self.init_cube()


    def init_cube(self):
        """
        Puts cube in default state
        """
        if "faces" in self.save_strategy:
            self.as_faces = self.faces_factory()
        if "cubies" in self.save_strategy:
            raise NotImplementedError("Cubies turns are not implemented yet")
            self.as_cubies = self.cubies_factory()
        if "turns" in self.save_strategy:
            self.as_turns = self.turns_factory()


    def scramble(self, n):
        """
        Randomly performs *n* turns on the cube, scrambling the cube
        """

        times: list[int] = [1, 2, 3] # clockwise turns; 1 - normal, 2 - 2, 3 - '
        banned = set()
        previous = ""
        for _ in range(n):
            weights = [0 if it in banned else 1 for it in consts.BASE_TURNS]
            turn: tuple[str, int] = random.choices(consts.BASE_TURNS, weights, k=1)[0], random.choice(times)
            self.turn(turn)
            if self.is_dependent(turn[0], previous):
                banned.clear()
            banned.add(turn[0])
            previous = turn[0]


    def generate_scrambles(self, n: int, specific: bool=True):
        """
        Generates and returns all scrambles as turns
        Args:
            n (int): number of 90 degree clockwise turns; -1 for infinite
            specific (bool): if True generates only scrambles with n 90 degree clockwise turns, if False, it also generates for <n. Defaults to True
        """
        assert isinstance(n, int) and n >= -1, "Invalid n"
        assert not specific or n > 0, "specific=True only works with finite n (not n = -1)"

        warnings.warn("Function generate_scrambles is very slow. Optimization is needed.")

        def is_valid(turns):
            if len(turns) == 0:
                return False
            elif len(turns) == 1:
                return True
            a, b = turns[0], turns[1]
            if a == b:
                return False
            for c in turns[2:]:
                if b == c or (a == c and self.is_dependent(a, b)):
                    return False
                a, b = b, c
            return True

        if not specific:
            i = 1
            while n >= i:
                yield from self.generate_scrambles(i, specific=True)
                i += 1

        else:
            for combo in itertools.product(range(18), repeat=n):
                turns = [(consts.BASE_TURNS[x // 3], x % 3 + 1) for x in combo]
                if not is_valid(list(zip(*turns))[0]):
                    continue
                yield turns


    def turn(self, turn):
        """
        Turns the cube
        Args:
            turn (tuple[str, int]): tuple (face, k), where face is one of [F, B, R, L, U, D] and k is one of [1, 2, 3]. For example (F, 1) represents one 90 degree clockwise turn of the front face.
        """
        face, k = turn
        if not face in consts.BASE_TURNS:
            raise ValueError("First item of tuple turn has to be one of [F, B, R, L, U, D]")
        if not k in [1, 2, 3]:
            raise ValueError("First item of tuple turn has to be one of [1, 2, 3]")

        if "turns" in self.save_strategy:
            self.as_turns.append(turn)

        if "faces" in self.save_strategy:
            for _ in range(k):
                as_faces_copy = copy.deepcopy(self.as_faces)

                face_idx = consts.BASE_TURNS.index(face)
                rot_indices = [(it-2)%8 for it in range(8)]
                self.as_faces[face_idx] = as_faces_copy[face_idx, rot_indices]

                turns = consts.POSITION_TURNS_MAP[face]
                for to_tuple, from_tuple in turns.items():
                    self.as_faces[*to_tuple] = as_faces_copy[*from_tuple]
        
        if "cubies" in self.save_strategy:
            raise NotImplementedError("Cubies turns are not implemented yet")
            for _ in range(k):
                turns = consts.CUBIES_TURNS_MAP[face]
                as_cubies_copy = copy.deepcopy(self.as_cubies)
                for from_idx, to_idx in turns.items():
                    self.as_cubies[to_idx] = as_cubies_copy[from_idx]
    

    def turn_set(self, turns: list[tuple[str, int]]):
        """
        Turns the cube with a set of turns
        Args:
            turns (list[tuple[str, int]]): list of turns to apply. See turn() for details on the turn format.
        """
        for turn in turns:
            self.turn(turn)


    def optimize_turns(self):
        """
        Optimizes turns
        """
        if "turns" not in self.save_strategy:
            return
        idx: int = 0
        while self.as_turns[idx + 1:]:
            turn: tuple[str, int] = self.as_turns[idx]
            matched_indexes = []
            for j, following in enumerate(self.as_turns[idx + 1:]):
                j += idx + 1
                if self.is_dependent(turn[0], following[0]) or j == len(self.as_turns)-1:
                    count = self.as_turns[idx][1]
                    for x in reversed(matched_indexes):
                        count += self.as_turns[x][1]
                        self.as_turns.pop(x)
                    count %= 4
                    if count == 0:
                        self.as_turns.pop(idx)
                    else:
                        self.as_turns[idx] = turn[0], count
                        idx += 1
                    break
                if turn[0] == following[0]:
                    matched_indexes.append(j)


    @staticmethod
    def is_dependent(turn1, turn2):
        """
        Checks if *turn1* and *turn2* change each other's pieces. In other words, if the order of these turns matters.
        """
        for pair in consts.INDEPENDENT_PAIRS:
            if turn1 in pair and turn2 in pair:
                return False
        return True


    def __str__(self):
        texts = []
        if "faces" in self.save_strategy:
            text = ""

            max_len = max(len(face) for face in consts.BASE_TURNS_FULL_NAME+consts.COLORS_FULL_NAME)
            base_turns = [it+" "*(max_len-len(it)) for it in consts.BASE_TURNS_FULL_NAME]
            color_names = [it+" "*(max_len-len(it)) for it in consts.COLORS_FULL_NAME]

            for arr, face in zip(self.as_faces, base_turns):
                text += f"{face} | {" | ".join(color_names[int(it)] for it in arr)}\n"

            texts.append(f"As faces:\n{text[:-1]}")
        if "cubies" in self.save_strategy:
            texts.append(f"As cubies: {self.as_cubies}")
        if "turns" in self.save_strategy:
            texts.append(f"As turns: {self.as_turns}")

        return "\n".join(texts)