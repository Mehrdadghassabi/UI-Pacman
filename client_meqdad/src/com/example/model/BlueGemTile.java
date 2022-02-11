package com.example.model;

public enum BlueGemTile {
    REWARD(75),
    REQUIRED_SCORE(140),
    MAXIMUM_ACHIEVABLE_TIMES(4);

    private final int value;

    BlueGemTile(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
