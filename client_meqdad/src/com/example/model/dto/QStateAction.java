package com.example.model.dto;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

@Builder
@Getter
@Setter
public class QStateAction implements Serializable {

    private byte x;
    private byte y;
    private byte action;
    private boolean[] gems;

    @Override
    public String toString() {
        return "QStateAction{" +
                "x=" + x +
                ", y=" + y +
                ", action=" + action +
                ", gems=" + Arrays.toString(gems) +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        QStateAction qStateAction = (QStateAction) o;
        return x == qStateAction.x &&
                y == qStateAction.y &&
                action == qStateAction.action &&
                Arrays.equals(gems, qStateAction.gems);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(x, y, action);
        result = 31 * result + Arrays.hashCode(gems);
        return result;
    }
}
