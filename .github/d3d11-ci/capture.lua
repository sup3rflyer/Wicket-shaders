-- Deterministic frame capture for cross-platform coherency checks. Starts paused,
-- warms the chain (so STORAGE/EMA leave cold-start), then steps one source frame
-- per tick and screenshots the PRESENTED window (redraw path — focus-independent,
-- so it works on a headless WARP runner). Deterministic / byte-reproducible, which
-- is why it's preferred over playing-mode timer capture for the SPATIAL metrics.
-- Caveat: a per-present reseed cadence (e.g. filmgrain-match's TPL jitter) is not
-- necessarily exercised by paused frame-step, so treat the TEMPORAL metric as
-- advisory, not a verdict. Env: CAP_FRAMES, CAP_WARM, CAP_DIR.
local N    = tonumber(os.getenv('CAP_FRAMES') or '24')
local WARM = tonumber(os.getenv('CAP_WARM')   or '8')
local DIR  = os.getenv('CAP_DIR') or 'cap'
local phase, wc, i, timer = 'warm', 0, 0, nil

local function tick()
    if phase == 'warm' then
        mp.commandv('frame-step')
        wc = wc + 1
        if wc >= WARM then phase = 'cap' end
        return
    end
    mp.commandv('screenshot-to-file', string.format('%s/%04d.png', DIR, i), 'window')
    i = i + 1
    if i >= N then
        if timer then timer:kill() end
        mp.command('quit')
        return
    end
    mp.commandv('frame-step')
end

mp.register_event('file-loaded', function()
    mp.add_timeout(0.6, function() timer = mp.add_periodic_timer(0.25, tick) end)
end)
