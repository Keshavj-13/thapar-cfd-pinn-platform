package com.thapar.cfd.controller;

import com.thapar.cfd.service.CfdEngineService;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.BinaryWebSocketHandler;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class CfdWebSocketHandler extends BinaryWebSocketHandler implements CfdEngineService.SliceConsumer {
    private static final Logger log = LoggerFactory.getLogger(CfdWebSocketHandler.class);
    private static final int MAGIC_HEADER = 0x54485052; // "THPR"

    private final Set<WebSocketSession> sessions = ConcurrentHashMap.newKeySet();
    private final CfdEngineService engineService;

    public CfdWebSocketHandler(CfdEngineService engineService) {
        this.engineService = engineService;
    }

    @PostConstruct
    public void init() {
        // Passive registration only -- does NOT start the compute loop. Live CFD is
        // strictly opt-in: the GPU only starts stepping when a user explicitly clicks
        // RUN / "Start Live Preview" (see CfdDashboardController#start).
        engineService.registerSliceConsumer(this);
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        sessions.add(session);
        log.info("Client connected to CFD binary stream: {}", session.getId());
        engineService.publishCurrentSlice();
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        sessions.remove(session);
        log.info("Client disconnected from CFD stream: {}", session.getId());
    }

    @Override
    public void onSliceAvailable(ByteBuffer floatBuffer, int width, int height, int axisId, String fieldName, float minVal, float maxVal) {
        if (sessions.isEmpty()) return;

        // Packet Structure:
        // [0..3]:   int32 magic (0x54485052)
        // [4..7]:   int32 width
        // [8..11]:  int32 height
        // [12..15]: int32 axisId (0=X, 1=Y, 2=Z) -- lets one socket multiplex simultaneous cut-planes
        // [16..19]: float32 minVal
        // [20..23]: float32 maxVal
        // [24..end]: raw float32 slice array (width * height * 4 bytes)
        int dataBytes = width * height * Float.BYTES;
        ByteBuffer packet = ByteBuffer.allocateDirect(24 + dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        packet.putInt(MAGIC_HEADER);
        packet.putInt(width);
        packet.putInt(height);
        packet.putInt(axisId);
        packet.putFloat(minVal);
        packet.putFloat(maxVal);

        floatBuffer.rewind();
        packet.put(floatBuffer);
        packet.flip();

        BinaryMessage msg = new BinaryMessage(packet);
        for (WebSocketSession session : sessions) {
            if (session.isOpen()) {
                try {
                    synchronized (session) {
                        session.sendMessage(msg);
                    }
                } catch (IOException e) {
                    log.warn("Failed to stream slice to session {}", session.getId());
                }
            }
        }
    }
}
