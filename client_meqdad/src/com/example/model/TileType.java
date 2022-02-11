package com.example.model;

public enum TileType {
    EMPTY("E"),
    TELEPORT("T"),
    WALL("W"),
    YELLOW_GEM("1"),
    GREEN_GEM("2"),
    RED_GEM("3"),
    BLUE_GEM("4");

    private final String value;

    TileType(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }
}
