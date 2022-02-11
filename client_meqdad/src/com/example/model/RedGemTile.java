package com.example.model;

public enum RedGemTile {
    REWARD(35),
    REQUIRED_SCORE(50),
    MAXIMUM_ACHIEVABLE_TIMES(5);

    private final int value;

    RedGemTile(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
