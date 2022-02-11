package com.example;

import com.example.logic.ImprovedReinforcementAnalyzer;
import com.example.logic.ReinforcementAnalyzer;
import com.example.model.Tile;
import com.example.model.TileType;
import com.example.model.dto.QStateAction;

import java.io.IOException;
import java.util.*;

public class Agent extends BaseAgent {

    private long timeout;
    private Set<Tile> blockedGems;
    private int trapUsed;
    private int remainedTrap;
    private ReinforcementAnalyzer reinforcementAnalyzer;
    private ImprovedReinforcementAnalyzer improvedReinforcementAnalyzer;
    private boolean[] gems;
    private List<Tile> gemsTiles;
    private boolean trainFlag;
    private boolean usingTrainedModel = true;

    public Agent() {
        super();
    }

    public Agent(String serverIp, int serverPort) {
        super(serverIp, serverPort);
        this.timeout = 1000000000;
        this.blockedGems = new HashSet<>();
        this.trapUsed = 0;
        this.remainedTrap = this.getTrapCount();
        this.reinforcementAnalyzer = new ReinforcementAnalyzer();
        this.improvedReinforcementAnalyzer = new ImprovedReinforcementAnalyzer();
        this.trainFlag = true;
    }

    public Agent(String serverIp) {
        super(serverIp);
    }

    public Agent(int serverPort) {
        super(serverPort);
    }

    @Override
    public Action doTurn() throws IOException {
        Utils utils = new Utils(this);
        long startTime = System.nanoTime();
//        if (trainFlag) {
            // training Q table
            if (this.getTurnCount() == 1)
                this.initGems();
            this.updateGemsVector();
            this.improvedReinforcementAnalyzer.updateParams(utils);
            this.improvedReinforcementAnalyzer.qStep();

            long secondTime = System.nanoTime();
            if (startTime - secondTime > this.timeout * 0.995)
                return Action.NoOp;

            if (this.getTurnCount() == this.getMaxTurnCount())
                this.improvedReinforcementAnalyzer.refreshParams();

            QStateAction qStateAction = QStateAction.builder()
                    .x((byte) utils.getMyAgentTile().getX())
                    .y((byte) utils.getMyAgentTile().getY())
                    .action((byte) -1)
                    .gems(this.gems.clone())
                    .build();

            return this.improvedReinforcementAnalyzer.action(qStateAction);
//        } else {
//            // using trained Q table
//            if (this.usingTrainedModel) {
//                System.out.println("using trained model ...!");
//                this.usingTrainedModel = false;
//            }
//            this.improvedReinforcementAnalyzer.setqTable(this.improvedReinforcementAnalyzer.getBestQTable());
//
//            QStateAction qStateAction = QStateAction.builder()
//                    .x((byte) utils.getMyAgentTile().getX())
//                    .y((byte) utils.getMyAgentTile().getY())
//                    .action((byte) -1)
//                    .gems(this.gems.clone())
//                    .build();
//            return this.improvedReinforcementAnalyzer.action(qStateAction);
//        }
    }

    public static void main(String[] args) throws IOException {
        var client = new Agent("127.0.0.1", 9921);
        client.play();
    }

    private void initGems() {
        List<Boolean> gems = new ArrayList<>();
        gemsTiles = new LinkedList<>();
        for (int i = 0; i < this.getGrid().length; i++) {
            for (int j = 0; j < this.getGrid()[0].length; j++) {
                switch (this.getGrid()[i][j]) {
                    case "1": {
                        Tile tile = new Tile(i, j, TileType.YELLOW_GEM);
                        gems.add(true);
                        gemsTiles.add(tile);
                        break;
                    }
                    case "2": {
                        Tile tile = new Tile(i, j, TileType.GREEN_GEM);
                        gems.add(true);
                        gemsTiles.add(tile);
                        break;
                    }
                    case "3": {
                        Tile tile = new Tile(i, j, TileType.RED_GEM);
                        gems.add(true);
                        gemsTiles.add(tile);
                        break;
                    }
                    case "4": {
                        Tile tile = new Tile(i, j, TileType.BLUE_GEM);
                        gems.add(true);
                        gemsTiles.add(tile);
                        break;
                    }
                }
            }
        }

        this.gems = new boolean[gems.size()];
        for (int i = 0; i < gems.size(); i++) {
            this.gems[i] = gems.get(i);
        }
    }

    private void updateGemsVector() {
        for (int i = 0; i < gemsTiles.size(); i++) {
            Tile tile = gemsTiles.get(i);
            this.gems[i] = this.isTileStillGem(tile);
        }
    }

    private boolean isTileStillGem(Tile tile) {
        int x = tile.getX();
        int y = tile.getY();
        return this.getGrid()[x][y].equals("1") || this.getGrid()[x][y].equals("2") ||
                this.getGrid()[x][y].equals("3") || this.getGrid()[x][y].equals("4");
    }

    public Set<Tile> getBlockedGems() {
        return blockedGems;
    }

    public void setBlockedGems(Set<Tile> blockedGems) {
        this.blockedGems = blockedGems;
    }

    public long getTimeout() {
        return timeout;
    }

    public void setTimeout(long timeout) {
        this.timeout = timeout;
    }

    public int getTrapUsed() {
        return trapUsed;
    }

    public void setTrapUsed(int trapUsed) {
        this.trapUsed = trapUsed;
    }

    public int getRemainedTrap() {
        return remainedTrap;
    }

    public void setRemainedTrap(int remainedTrap) {
        this.remainedTrap = remainedTrap;
    }

    public boolean[] getGems() {
        return gems;
    }

    public void setGems(boolean[] gems) {
        this.gems = gems;
    }

    public boolean isTrainFlag() {
        return trainFlag;
    }

    public void setTrainFlag(boolean trainFlag) {
        this.trainFlag = trainFlag;
    }
}
