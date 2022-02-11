package com.example.logic;

import com.example.BaseAgent;
import com.example.Utils;
import com.example.model.Tile;
import com.example.model.TileType;
import com.example.model.dto.ComplexBFSNode;
import com.example.model.dto.QStateAction;

import java.io.*;
import java.util.*;

public class ImprovedReinforcementAnalyzer {

    private Utils utils;
    private static float EPSILON = 0.9f;
    private final static float EPSILON_DECAY = 0.98f;
    private final static float LEARNING_RATE = 0.1f;
    private final static float DISCOUNT_FACTOR = 0.95f;
    private HashMap<QStateAction, Float> qTable;
    private HashMap<Byte, BaseAgent.Action> actionValueMap;
    private int lastAgentScore;
    private Tile lastAgentPosition;
    private BaseAgent.Action lastAgentAction;
    private boolean[] lastAgentGemsVector;
    private int rewards = 0;
    private HashMap<QStateAction, Float> bestQTable;
    private double bestAvgReward = Integer.MIN_VALUE;
    private int episodes = 0;
    private final static int STOP_TRAINING_THRESHOLD = 500;

    public void updateParams(Utils utils) {
        this.utils = utils;
        this.initActionValueMap();
        this.initQTable();
    }

    public void refreshParams() {
        this.lastAgentScore = 0;
        this.lastAgentPosition = null;
        this.lastAgentGemsVector = null;
    }

    public void qStep() throws IOException {
        if (this.lastAgentPosition == null) {
            this.rewards = 0;
            this.lastAgentScore = this.utils.getMyAgentScore();
            this.lastAgentPosition = this.utils.getMyAgentTile();
            this.lastAgentGemsVector = this.utils.getAgent().getGems().clone();
            return;
        }

        byte score = (byte) this.lastAgentScore;
        byte newScore = (byte) this.utils.getMyAgentScore();
        int reward = (newScore - score);

        Tile observation = this.lastAgentPosition;
        Tile newObservation = this.utils.getMyAgentTile();

        boolean[] gemsVector = this.lastAgentGemsVector;
        boolean[] newGemsVector = this.utils.getAgent().getGems().clone();

        if (this.allGemsPicked()) {
            if (this.utils.getAgent().getTurnCount() == this.utils.getAgent().getMaxTurnCount()) {
                this.episodes++;
                if (this.episodes > STOP_TRAINING_THRESHOLD) {
                    this.utils.getAgent().setTrainFlag(false);
                }
                double avgReward = (float) this.rewards / (float) this.utils.getAgent().getMaxTurnCount();
                if (this.bestAvgReward < avgReward) {
                    this.bestAvgReward = avgReward;
                    this.bestQTable = this.utils.qTableDeepCopy(this.qTable);
                    this.writeQTableToFile(this.bestQTable);
                }
                System.out.println("average reward for this game was: " + avgReward);
                System.out.println("best average reward until here: " + this.bestAvgReward);
            }
            return;
        }

        byte action = this.getActionValueByActionName(this.lastAgentAction);
        QStateAction qStateAction = QStateAction.builder()
                .x((byte) observation.getX())
                .y((byte) observation.getY())
                .action(action)
                .gems(gemsVector)
                .build();
        float currentQ = this.qTable.get(qStateAction);

        QStateAction newQStateAction = QStateAction.builder()
                .x((byte) newObservation.getX())
                .y((byte) newObservation.getY())
                .action((byte) -1)
                .gems(newGemsVector)
                .build();
        byte newAction = this.getActionWithMaxReward(newQStateAction);
        if (newAction == -1) {
            newAction = this.getRandomAction(qStateAction);
        }
        newQStateAction.setAction(newAction);

        float maxFutureQ = this.qTable.getOrDefault(newQStateAction, 0.0f);

//        short positionalReward = this.getPositionalReward(this.lastAgentPosition);
//        reward += positionalReward;

        float newQ;
        if (reward == -1) {
            reward = reward * this.utils.getAgent().getTurnCount();
        }
        newQ = currentQ + LEARNING_RATE * (reward + DISCOUNT_FACTOR * maxFutureQ - currentQ);

        this.qTable.put(qStateAction, newQ);
        this.lastAgentScore = newScore;
        this.lastAgentPosition = newObservation;
        this.lastAgentGemsVector = newGemsVector;
//        System.out.println(qStateAction + " -> " + newQ);
        this.rewards += reward;
        if (this.utils.getAgent().getTurnCount() == this.utils.getAgent().getMaxTurnCount()) {
            this.episodes++;
            if (this.episodes > STOP_TRAINING_THRESHOLD) {
                this.utils.getAgent().setTrainFlag(false);
            }
            double avgReward = (float) this.rewards / (float) this.utils.getAgent().getMaxTurnCount();
            if (this.bestAvgReward < avgReward) {
                this.bestAvgReward = avgReward;
                this.bestQTable = this.utils.qTableDeepCopy(this.qTable);
                this.writeQTableToFile(this.bestQTable);
            }
            System.out.println("average reward for this game was: " + avgReward);
            System.out.println("best average reward until here: " + this.bestAvgReward);
        }
    }

    public BaseAgent.Action action(QStateAction qStateAction) {
        byte actionValue;
        if (Math.random() > EPSILON) {
            actionValue = this.getActionWithMaxReward(qStateAction);
            if (actionValue == -1) {
                actionValue = this.getRandomAction(qStateAction);
            }
        } else {
            actionValue = this.getRandomAction(qStateAction);
        }

        BaseAgent.Action action = this.actionValueMap.get(actionValue);
        this.lastAgentAction = action;
        qStateAction.setAction(actionValue);
        this.qTable.put(qStateAction, 0.0f);
        EPSILON = EPSILON * EPSILON_DECAY;
        return action;
    }

    private boolean allGemsPicked() {
        for (boolean bool : this.lastAgentGemsVector) {
            if (bool)
                return false;
        }

        return true;
    }

    private short getPositionalReward(Tile source) {
        byte MAX_GEMS_TO_EXPLORE = 3;
        List<Tile> nearestGemsToExplore = this.utils.BFSToGetTopNearestReachableGems(source, MAX_GEMS_TO_EXPLORE,
                this.utils.getAgent().getGrid(), this.utils.getMyAgentCollectedGemsMap(), this.utils.getMyAgentScore());
        ComplexBFSNode result = this.utils.getOptimalOrder(source, nearestGemsToExplore, this.utils.getAgent().getGrid(),
                this.utils.getAgent().getTurnCount(), this.utils.getMyAgentScore());

        return (short) result.getRewardUntilHere();
    }

    private byte getRandomAction(QStateAction qStateAction) {
        List<Byte> possibleActions = this.getPossibleActions(qStateAction);
        Random random = new Random();
        int possibleActionsSize = possibleActions.size();
        int randomIndex = random.nextInt(possibleActionsSize);
        return possibleActions.get(randomIndex);
    }

    private byte getActionWithMaxReward(QStateAction qStateAction) {
        List<Byte> possibleActions = this.getPossibleActions(qStateAction);
        byte maxActionValue = -1;
        float max = Float.MIN_VALUE;
        for (Byte possibleAction : possibleActions) {
            QStateAction tempQStateAction = QStateAction.builder()
                    .x(qStateAction.getX())
                    .y(qStateAction.getY())
                    .action(possibleAction)
                    .gems(qStateAction.getGems().clone())
                    .build();
            if (this.qTable.containsKey(tempQStateAction)) {
                float value = this.qTable.get(tempQStateAction);
                if (value > max) {
                    max = value;
                    maxActionValue = tempQStateAction.getAction();
                }
            }
        }

        return maxActionValue;
    }

    private List<Byte> getPossibleActions(QStateAction qStateAction) {
        byte x = qStateAction.getX();
        byte y = qStateAction.getY();
        TileType type = this.utils.getTileType(x, y, this.utils.getAgent().getGrid());
        Tile tile = new Tile(x, y, type);
        List<Byte> possibleActions = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            if (this.isActionPossible((byte) i, this.utils.getAgent().getGrid(), tile)) {
                possibleActions.add((byte) i);
            }
        }

        return possibleActions;
    }

    private void initQTable() {
        if (this.qTable == null)
            this.qTable = new HashMap<>();
        if (this.bestQTable == null) {
            this.bestQTable = new HashMap<>();
        }
    }

    private void initActionValueMap() {
        if (this.actionValueMap != null)
            return;
        this.actionValueMap = new HashMap<>();
        for (int i = 0; i < BaseAgent.Action.values().length; i++) {
            BaseAgent.Action action = BaseAgent.Action.values()[i];
            if (action.equals(BaseAgent.Action.Up))
                this.actionValueMap.put((byte) 0, action);
            else if (action.equals(BaseAgent.Action.Down))
                this.actionValueMap.put((byte) 1, action);
            else if (action.equals(BaseAgent.Action.Right))
                this.actionValueMap.put((byte) 2, action);
            else if (action.equals(BaseAgent.Action.Left))
                this.actionValueMap.put((byte) 3, action);
            else if (action.equals(BaseAgent.Action.Teleport))
                this.actionValueMap.put((byte) 4, action);
        }
    }

    private boolean isActionPossible(byte actionValue, String[][] grid, Tile tile) {
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

    private byte getActionValueByActionName(BaseAgent.Action action) {
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

    private void writeQTableToFile(HashMap<QStateAction, Float> map) throws IOException {
        File file = new File("BestQTable.txt");
        FileOutputStream fos = new FileOutputStream(file);
        PrintWriter pw = new PrintWriter(fos);

        for(Map.Entry<QStateAction, Float> m :map.entrySet()){
            pw.println(m.getKey() + " = " + m.getValue());
        }

        pw.flush();
        pw.close();
        fos.close();
    }

    public HashMap<QStateAction, Float> getqTable() {
        return qTable;
    }

    public void setqTable(HashMap<QStateAction, Float> qTable) {
        this.qTable = qTable;
    }

    public HashMap<QStateAction, Float> getBestQTable() {
        return bestQTable;
    }

    public void setBestQTable(HashMap<QStateAction, Float> bestQTable) {
        this.bestQTable = bestQTable;
    }
}
