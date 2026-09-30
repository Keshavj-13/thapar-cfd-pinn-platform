package com.thapar.cfd.service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Minimal, sandboxed math expression parser/evaluator for the Custom Equations
 * feature (hand-rolled recursive descent -- no eval/reflection/scripting
 * engine, so there is no code-injection surface). Supports +,-,*,/,^, unary
 * minus, parentheses, numeric literals (incl. scientific notation), variables
 * looked up in the supplied map, the constants pi/e, and a fixed whitelist of
 * math functions.
 */
public final class ExpressionEvaluator {
    private final String src;
    private int pos;
    private final Map<String, Double> vars;

    private ExpressionEvaluator(String src, Map<String, Double> vars) {
        this.src = src;
        this.vars = vars;
        this.pos = 0;
    }

    public static double evaluate(String expression, Map<String, Double> vars) {
        if (expression == null || expression.isBlank()) {
            throw new IllegalArgumentException("Expression is empty");
        }
        ExpressionEvaluator ev = new ExpressionEvaluator(expression, vars);
        double v = ev.parseExpr();
        ev.skipWs();
        if (ev.pos != ev.src.length()) {
            throw new IllegalArgumentException("Unexpected character at position " + ev.pos + ": '" + ev.src.charAt(ev.pos) + "'");
        }
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            throw new IllegalArgumentException("Expression evaluated to a non-finite value (check for division by zero or a domain error such as sqrt of a negative number)");
        }
        return v;
    }

    private void skipWs() {
        while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) pos++;
    }

    private char peek() {
        skipWs();
        return pos < src.length() ? src.charAt(pos) : '\0';
    }

    private boolean consume(char c) {
        if (peek() == c) { pos++; return true; }
        return false;
    }

    private double parseExpr() {
        double v = parseTerm();
        while (true) {
            char c = peek();
            if (c == '+') { pos++; v += parseTerm(); }
            else if (c == '-') { pos++; v -= parseTerm(); }
            else break;
        }
        return v;
    }

    private double parseTerm() {
        double v = parseUnary();
        while (true) {
            char c = peek();
            if (c == '*') { pos++; v *= parseUnary(); }
            else if (c == '/') { pos++; v /= parseUnary(); }
            else break;
        }
        return v;
    }

    private double parseUnary() {
        if (consume('-')) return -parseUnary();
        if (consume('+')) return parseUnary();
        return parsePower();
    }

    private double parsePower() {
        double base = parseAtom();
        if (consume('^')) {
            double exp = parseUnary(); // right-associative
            return Math.pow(base, exp);
        }
        return base;
    }

    private double parseAtom() {
        char c = peek();
        if (c == '(') {
            pos++;
            double v = parseExpr();
            if (!consume(')')) throw new IllegalArgumentException("Missing closing ')'");
            return v;
        }
        if (Character.isDigit(c) || c == '.') {
            return parseNumber();
        }
        if (Character.isLetter(c) || c == '_') {
            return parseIdentOrCall();
        }
        throw new IllegalArgumentException("Unexpected character at position " + pos + (pos < src.length() ? ": '" + src.charAt(pos) + "'" : " (end of input)"));
    }

    private double parseNumber() {
        skipWs();
        int start = pos;
        while (pos < src.length() && (Character.isDigit(src.charAt(pos)) || src.charAt(pos) == '.')) pos++;
        if (pos < src.length() && (src.charAt(pos) == 'e' || src.charAt(pos) == 'E')) {
            int save = pos;
            pos++;
            if (pos < src.length() && (src.charAt(pos) == '+' || src.charAt(pos) == '-')) pos++;
            if (pos < src.length() && Character.isDigit(src.charAt(pos))) {
                while (pos < src.length() && Character.isDigit(src.charAt(pos))) pos++;
            } else {
                pos = save;
            }
        }
        return Double.parseDouble(src.substring(start, pos));
    }

    private double parseIdentOrCall() {
        skipWs();
        int start = pos;
        while (pos < src.length() && (Character.isLetterOrDigit(src.charAt(pos)) || src.charAt(pos) == '_')) pos++;
        String name = src.substring(start, pos);
        if (peek() == '(') {
            pos++;
            List<Double> args = new ArrayList<>();
            if (peek() != ')') {
                args.add(parseExpr());
                while (consume(',')) args.add(parseExpr());
            }
            if (!consume(')')) throw new IllegalArgumentException("Missing closing ')' in call to " + name);
            return applyFunction(name, args);
        }
        return resolveIdent(name);
    }

    private double resolveIdent(String name) {
        switch (name) {
            case "pi": return Math.PI;
            case "e": return Math.E;
        }
        Double v = vars.get(name);
        if (v == null) {
            throw new IllegalArgumentException("Unknown variable '" + name + "' (available: x, y, z, xn, yn, zn, lx, ly, lz, pi, e)");
        }
        return v;
    }

    private double applyFunction(String name, List<Double> a) {
        switch (name) {
            case "sin": require(a, 1, name); return Math.sin(a.get(0));
            case "cos": require(a, 1, name); return Math.cos(a.get(0));
            case "tan": require(a, 1, name); return Math.tan(a.get(0));
            case "exp": require(a, 1, name); return Math.exp(a.get(0));
            case "sqrt": require(a, 1, name); return Math.sqrt(a.get(0));
            case "abs": require(a, 1, name); return Math.abs(a.get(0));
            case "log": require(a, 1, name); return Math.log(a.get(0));
            case "floor": require(a, 1, name); return Math.floor(a.get(0));
            case "ceil": require(a, 1, name); return Math.ceil(a.get(0));
            case "step": require(a, 1, name); return a.get(0) >= 0 ? 1.0 : 0.0;
            case "pow": require(a, 2, name); return Math.pow(a.get(0), a.get(1));
            case "min": require(a, 2, name); return Math.min(a.get(0), a.get(1));
            case "max": require(a, 2, name); return Math.max(a.get(0), a.get(1));
            case "clamp": require(a, 3, name); return Math.max(a.get(1), Math.min(a.get(2), a.get(0)));
            default:
                throw new IllegalArgumentException("Unknown function '" + name + "' (available: sin, cos, tan, exp, sqrt, abs, log, floor, ceil, step, pow, min, max, clamp)");
        }
    }

    private void require(List<Double> a, int n, String name) {
        if (a.size() != n) {
            throw new IllegalArgumentException(name + "() expects " + n + " argument" + (n == 1 ? "" : "s") + ", got " + a.size());
        }
    }
}
