(function () {
    if (window.__fpHooksInstalled) return;
    window.__fpHooksInstalled = true;
    window.__fpCalls = [];
    const _FP_CALLS_CAP = 5000;

    // Normaliza propriedades do browser para que ambos os engines recebam o mesmo
    // bundle JS e executem o mesmo código-fonte (necessário para alinhar a IR).
    if (typeof window.chrome === 'undefined') {
        Object.defineProperty(window, 'chrome', {
            value: { runtime: {} },
            configurable: true, writable: true,
        });
    }
    if (navigator.vendor === '') {
        try {
            Object.defineProperty(navigator, 'vendor', { get: () => 'Google Inc.' });
        } catch (_) {}
    }
    // Remove APIs exclusivas do Firefox que bibliotecas FP usam para detectar o
    // engine e selecionar code paths diferentes (ex: fingerprintjs pula o canvas
    // fingerprint se detectar Firefox via buildID, MozAppearance, etc.).
    const _firefoxNavProps = ['buildID', 'globalPrivacyControl', 'oscpu'];
    for (const prop of _firefoxNavProps) {
        if (prop in navigator) {
            try { delete Navigator.prototype[prop]; } catch (_) {}
            try { delete navigator[prop]; } catch (_) {}
        }
    }
    const _firefoxWinProps = ['mozInnerScreenX', 'onmozfullscreenchange', 'CSSMozDocumentRule'];
    for (const prop of _firefoxWinProps) {
        if (prop in window) {
            try { delete window[prop]; } catch (_) {}
        }
    }
    try {
        const s = document.documentElement.style;
        if ('MozAppearance' in s) delete s.MozAppearance;
    } catch (_) {}

    function parseFrame(frame) {
        // V8/Chrome: "    at funcName (url:line:col)"  ou  "    at url:line:col"
        const v8Named = frame.match(/at ([^(@]+)\s+\((.+):(\d+):(\d+)\)/);
        const v8Anon  = frame.match(/^\s+at (https?:\/\/.+):(\d+):(\d+)\)?$/);
        // Gecko/Firefox: "funcName@url:line:col"  ou  "@url:line:col"
        const geckoNamed = frame.match(/^([^@\s]+)@(https?:\/\/.+):(\d+):(\d+)$/);
        const geckoAnon  = frame.match(/^@(https?:\/\/.+):(\d+):(\d+)$/);

        if (v8Named)    return { funcName: v8Named[1].trim(),    scriptUrl: v8Named[2],    line: +v8Named[3],    col: +v8Named[4]    };
        if (geckoNamed) return { funcName: geckoNamed[1].trim(), scriptUrl: geckoNamed[2], line: +geckoNamed[3], col: +geckoNamed[4] };
        if (v8Anon)     return { funcName: 'anonymous',          scriptUrl: v8Anon[1],     line: +v8Anon[2],    col: +v8Anon[3]    };
        if (geckoAnon)  return { funcName: 'anonymous',          scriptUrl: geckoAnon[1],  line: +geckoAnon[2], col: +geckoAnon[3] };
        return null;
    }

    // Captura o stack trace ignorando qualquer Error.prepareStackTrace customizado
    // que bibliotecas de fingerprinting (ex: creepjs) possam ter instalado.
    function getRawStack() {
        const saved = Error.prepareStackTrace;
        Error.prepareStackTrace = undefined;
        const err = new Error();
        Error.prepareStackTrace = saved;
        return err.stack || '';
    }

    // Emite uma entrada em __fpCalls para CADA frame HTTP único no call stack.
    // Isso atribui a chamada a todas as funções presentes na cadeia de execução,
    // não apenas ao caller imediato — necessário para que heurísticas multi-critério
    // (ex: audio FP = createOscillator + startRendering + getChannelData na mesma
    // função) funcionem mesmo quando os sub-calls vêm de funções internas.
    function capture(api, arg) {
        if (window.__fpCalls.length >= _FP_CALLS_CAP) return;
        const stack = getRawStack();
        const lines = stack.split('\n');

        const seen = new Set();
        for (const line of lines) {
            if (!line.includes('http://') && !line.includes('https://')) continue;
            const result = parseFrame(line.trim());
            if (!result) continue;
            const key = result.funcName + '|' + result.scriptUrl + '|' + result.line;
            if (seen.has(key)) continue;
            seen.add(key);
            const entry = { api, funcName: result.funcName, scriptUrl: result.scriptUrl,
                            line: result.line, col: result.col };
            if (arg !== undefined) entry.arg = String(arg).slice(0, 256);
            window.__fpCalls.push(entry);
        }
    }

    // ── Canvas fingerprinting ─────────────────────────────────────────────────
    const _toDataURL = HTMLCanvasElement.prototype.toDataURL;
    HTMLCanvasElement.prototype.toDataURL = function () {
        capture('canvas.toDataURL');
        return _toDataURL.apply(this, arguments);
    };

    const _getImageData = CanvasRenderingContext2D.prototype.getImageData;
    CanvasRenderingContext2D.prototype.getImageData = function () {
        capture('canvas.getImageData');
        return _getImageData.apply(this, arguments);
    };

    // fillText: arg = texto renderizado (comprimento ≥ 10 é critério de canvas FP)
    const _fillText = CanvasRenderingContext2D.prototype.fillText;
    CanvasRenderingContext2D.prototype.fillText = function (text) {
        capture('canvas.fillText', text);
        return _fillText.apply(this, arguments);
    };

    // Critérios negativos: presença de save/restore/addEventListener indica uso
    // geral de canvas (não fingerprinting específico).
    const _save = CanvasRenderingContext2D.prototype.save;
    CanvasRenderingContext2D.prototype.save = function () {
        capture('canvas.save');
        return _save.apply(this, arguments);
    };

    const _restore = CanvasRenderingContext2D.prototype.restore;
    CanvasRenderingContext2D.prototype.restore = function () {
        capture('canvas.restore');
        return _restore.apply(this, arguments);
    };

    try {
        const _ctxAddEventListener = CanvasRenderingContext2D.prototype.addEventListener;
        if (_ctxAddEventListener) {
            CanvasRenderingContext2D.prototype.addEventListener = function () {
                capture('canvas.addEventListener');
                return _ctxAddEventListener.apply(this, arguments);
            };
        }
    } catch (_) {}

    // ── Canvas Font fingerprinting ────────────────────────────────────────────
    const _measureText = CanvasRenderingContext2D.prototype.measureText;
    CanvasRenderingContext2D.prototype.measureText = function () {
        capture('canvas.measureText');
        return _measureText.apply(this, arguments);
    };

    // font setter: arg = valor da fonte (heurística requer > 20 valores distintos)
    try {
        const _fontDesc = Object.getOwnPropertyDescriptor(CanvasRenderingContext2D.prototype, 'font');
        if (_fontDesc && _fontDesc.set) {
            Object.defineProperty(CanvasRenderingContext2D.prototype, 'font', {
                get: _fontDesc.get,
                set: function (value) {
                    capture('canvas.font', value);
                    return _fontDesc.set.call(this, value);
                },
                configurable: true,
            });
        }
    } catch (_) {}

    // ── OffscreenCanvas (mesmas heurísticas, API paralela) ────────────────────
    try {
        if (window.OffscreenCanvasRenderingContext2D) {
            const _oFillText = OffscreenCanvasRenderingContext2D.prototype.fillText;
            OffscreenCanvasRenderingContext2D.prototype.fillText = function (text) {
                capture('canvas.fillText', text);
                return _oFillText.apply(this, arguments);
            };
            const _oMeasure = OffscreenCanvasRenderingContext2D.prototype.measureText;
            OffscreenCanvasRenderingContext2D.prototype.measureText = function () {
                capture('canvas.measureText');
                return _oMeasure.apply(this, arguments);
            };
            const _oFontDesc = Object.getOwnPropertyDescriptor(OffscreenCanvasRenderingContext2D.prototype, 'font');
            if (_oFontDesc && _oFontDesc.set) {
                Object.defineProperty(OffscreenCanvasRenderingContext2D.prototype, 'font', {
                    get: _oFontDesc.get,
                    set: function (value) { capture('canvas.font', value); return _oFontDesc.set.call(this, value); },
                    configurable: true,
                });
            }
        }
    } catch (_) {}

    // ── AudioContext fingerprinting ───────────────────────────────────────────
    try {
        const _createOscillator = BaseAudioContext.prototype.createOscillator;
        BaseAudioContext.prototype.createOscillator = function () {
            capture('audio.createOscillator');
            return _createOscillator.apply(this, arguments);
        };
    } catch (_) {}

    try {
        const _createDynamicsCompressor = BaseAudioContext.prototype.createDynamicsCompressor;
        BaseAudioContext.prototype.createDynamicsCompressor = function () {
            capture('audio.createDynamicsCompressor');
            return _createDynamicsCompressor.apply(this, arguments);
        };
    } catch (_) {}

    try {
        const _startRendering = OfflineAudioContext.prototype.startRendering;
        OfflineAudioContext.prototype.startRendering = function () {
            capture('audio.startRendering');
            return _startRendering.apply(this, arguments);
        };
    } catch (_) {}

    try {
        const _connect = AudioNode.prototype.connect;
        AudioNode.prototype.connect = function () {
            capture('audio.connect');
            return _connect.apply(this, arguments);
        };
    } catch (_) {}

    const _copyFromChannel = AudioBuffer.prototype.copyFromChannel;
    AudioBuffer.prototype.copyFromChannel = function () {
        capture('audio.copyFromChannel');
        return _copyFromChannel.apply(this, arguments);
    };

    try {
        const _getChannelData = AudioBuffer.prototype.getChannelData;
        AudioBuffer.prototype.getChannelData = function () {
            capture('audio.getChannelData');
            return _getChannelData.apply(this, arguments);
        };
    } catch (_) {}

    // ── WebGL fingerprinting ──────────────────────────────────────────────────
    // getParameter with RENDERER/VENDOR constants is exclusively a FP signal.
    try {
        const _getParamWGL = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function (pname) {
            capture('webgl.getParameter', pname);
            return _getParamWGL.call(this, pname);
        };
    } catch (_) {}
    try {
        const _getParamWGL2 = WebGL2RenderingContext.prototype.getParameter;
        WebGL2RenderingContext.prototype.getParameter = function (pname) {
            capture('webgl.getParameter', pname);
            return _getParamWGL2.call(this, pname);
        };
    } catch (_) {}
    try {
        const _getSupportedExt = WebGLRenderingContext.prototype.getSupportedExtensions;
        WebGLRenderingContext.prototype.getSupportedExtensions = function () {
            capture('webgl.getSupportedExtensions');
            return _getSupportedExt.call(this);
        };
    } catch (_) {}

    // ── Screen fingerprinting ─────────────────────────────────────────────────
    // colorDepth/pixelDepth are never read for rendering — pure FP signal.
    try {
        const _screenProps = ['colorDepth', 'pixelDepth', 'width', 'height', 'availWidth'];
        for (const prop of _screenProps) {
            const desc = Object.getOwnPropertyDescriptor(Screen.prototype, prop);
            if (desc && desc.get) {
                Object.defineProperty(Screen.prototype, prop, {
                    get: function () { capture('screen.' + prop); return desc.get.call(this); },
                    configurable: true,
                });
            }
        }
    } catch (_) {}

    // ── Navigator fingerprinting ──────────────────────────────────────────────
    // High Entropy navigator properties per Chromium's privacy budget list.
    try {
        const _navProps = ['hardwareConcurrency', 'languages', 'platform', 'deviceMemory', 'maxTouchPoints'];
        for (const prop of _navProps) {
            const desc = Object.getOwnPropertyDescriptor(Navigator.prototype, prop);
            if (desc && desc.get) {
                Object.defineProperty(Navigator.prototype, prop, {
                    get: function () { capture('navigator.' + prop); return desc.get.call(this); },
                    configurable: true,
                });
            }
        }
    } catch (_) {}

    // ── WebRTC fingerprinting ─────────────────────────────────────────────────
    const _createOffer = RTCPeerConnection.prototype.createOffer;
    RTCPeerConnection.prototype.createOffer = function () {
        capture('webrtc.createOffer');
        return _createOffer.apply(this, arguments);
    };

    try {
        const _createDataChannel = RTCPeerConnection.prototype.createDataChannel;
        RTCPeerConnection.prototype.createDataChannel = function () {
            capture('webrtc.createDataChannel');
            return _createDataChannel.apply(this, arguments);
        };
    } catch (_) {}

    try {
        const _setLocalDescription = RTCPeerConnection.prototype.setLocalDescription;
        RTCPeerConnection.prototype.setLocalDescription = function () {
            capture('webrtc.setLocalDescription');
            return _setLocalDescription.apply(this, arguments);
        };
    } catch (_) {}
})();
