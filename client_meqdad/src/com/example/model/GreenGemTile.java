package com.example.model;

public enum GreenGemTile {
    REWARD(25),
    REQUIRED_SCORE(15),
    MAXIMUM_ACHIEVABLE_TIMES(8);

    private final int value;

    GreenGemTile(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
