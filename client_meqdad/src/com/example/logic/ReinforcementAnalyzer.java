package com.example.logic;

import com.example.BaseAgent;
import com.example.Utils;
import com.example.model.Tile;
import com.example.model.TileType;

import java.util.*;

public class ReinforcementAnalyzer {

    private Utils utils;
    private final static double EPSILON = 0.9;
    private final static double EPSILON_DECAY = 0.9998;
    private final static double LEARNING_RATE = 0.1;
    private final static double DISCOUNT_FACTOR = 0.95;
    private final static int NUMBER_OF_ACTIONS = 5;
    private double[][][] qTable;
    private HashMap<Integer, BaseAgent.Action> actionValueMap;
    private final LinkedList<Integer> agentScores = new LinkedList<>();
    private final LinkedList<Tile> agentPositions = new LinkedList<>();
    private final LinkedList<BaseAgent.Action> agentActions = new LinkedList<>();

    public void updateParams(Utils utils) {
        this.utils = utils;
        this.initActionValueMap();
        this.initQTable();
    }

    public void qStep() {
        if (this.agentScores.isEmpty()) {
            this.agentScores.add(this.utils.getMyAgentScore());
            this.agentPositions.add(this.utils.getMyAgentTile());
            return;
        }
        int score = this.agentScores.getLast();
        int newScore = this.utils.getMyAgentScore();
        int reward = newScore - score;

        Tile observation = this.agentPositions.getLast();
        Tile newObservation = this.utils.getMyAgentTile();

        int maxFutureActionIndex = this.getActionWithMaxReward(this.qTable[newObservation.getX()][newObservation.getY()]);
        double maxFutureQ = this.qTable[newObservation.getX()][newObservation.getY()][maxFutureActionIndex];

        int action = this.getActionValueByActionName(this.agentActions.getLast());
        double currentQ = this.qTable[observation.getY()][observation.getY()][action];

        double newQ;
        if (reward != -1) {
            newQ = reward;
        } else {
            newQ = currentQ + LEARNING_RATE * (reward + DISCOUNT_FACTOR * maxFutureQ - currentQ);
        }

        this.qTable[observation.getX()][observation.getY()][action] = newQ;
        this.agentScores.add(newScore);
        this.agentPositions.add(newObservation);
        System.out.println(Arrays.deepToString(this.qTable));
    }

    public BaseAgent.Action action(int x, int y) {
        int actionValue;
        double[] channels = this.qTable[x][y];
        if (Math.random() > EPSILON) {
            actionValue = this.getActionWithMaxReward(channels);
        } else {
            ArrayList<Integer> possibleActions = this.getPossibleActions(channels);
            Random random = new Random();
            int possibleActionsSize = possibleActions.size();
            int randomIndex = random.nextInt(possibleActionsSize);
            actionValue = possibleActions.get(randomIndex);
        }

        BaseAgent.Action action = this.actionValueMap.get(actionValue);
        this.agentActions.add(action);
        return action;
    }

    private int getActionWithMaxReward(double... channels) {
        double max = Integer.MIN_VALUE;
        int maxIndex = -1;
        for (int i = 0; i < channels.length; i++) {
            if (channels[i] > max) {
                max = channels[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    private ArrayList<Integer> getPossibleActions(double... channels) {
        ArrayList<Integer> possibleActions = new ArrayList<>();
        for (int i = 0; i < channels.length; i++) {
            if (channels[i] != Integer.MIN_VALUE) {
                possibleActions.add(i);
            }
        }

        return possibleActions;
    }

    private void initActionValueMap() {
        if (this.actionValueMap != null)
            return;
        this.actionValueMap = new HashMap<>();
        for (int i = 0; i < BaseAgent.Action.values().length; i++) {
            BaseAgent.Action action = BaseAgent.Action.values()[i];
            if (action.equals(BaseAgent.Action.Up))
                this.actionValueMap.put(0, action);
            else if (action.equals(BaseAgent.Action.Down))
                this.actionValueMap.put(1, action);
            else if (action.equals(BaseAgent.Action.Right))
                this.actionValueMap.put(2, action);
            else if (action.equals(BaseAgent.Action.Left))
                this.actionValueMap.put(3, action);
            else if (action.equals(BaseAgent.Action.Teleport))
                this.actionValueMap.put(4, action);
        }
    }

    private void initQTable() {
        if (this.qTable != null)
            return;
        int height = this.utils.getAgent().getGridHeight();
        int width = this.utils.getAgent().getGridWidth();
        this.qTable = new double[height][width][NUMBER_OF_ACTIONS];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < NUMBER_OF_ACTIONS; k++) {
                    TileType type = this.utils.getTileType(i, j, this.utils.getAgent().getGrid());
                    Tile tile = new Tile(i, j, type);
                    if (this.isActionPossible(k, this.utils.getAgent().getGrid(), tile)) {
                        this.qTable[i][j][k] = 0;
                    } else
                        this.qTable[i][j][k] = Integer.MIN_VALUE;
                }
            }
        }
    }

    private boolean isActionPossible(int actionValue, String[][] grid, Tile tile) {
        if (actionValue == 0) {
            return this.utils.isUpActionValid(tile, grid);
        } else if (actionValue == 1) {
            return this.utils.isDownActionValid(tile, grid);
        } else if (actionValue == 2) {
            return this.utils.isRightActionValid(tile, grid);
        } else if (actionValue == 3) {
            return this.utils.isLeftActionValid(tile, grid);
        } else if (actionValue == 4) {
            return this.utils.isTeleportActionValid(tile, grid);
        } else throw new IllegalStateException("action value is invalid!");
    }

    private int getActionValueByActionName(BaseAgent.Action action) {
        if (action.equals(BaseAgent.Action.Up))
            return 0;
        else if (action.equals(BaseAgent.Action.Down))
            return 1;
        else if (action.equals(BaseAgent.Action.Right))
            return 2;
        else if (action.equals(BaseAgent.Action.Left))
            return 3;
        else if (action.equals(BaseAgent.Action.Teleport))
            return 4;
        else
            throw new IllegalStateException("action is not valid!");
    }
}
