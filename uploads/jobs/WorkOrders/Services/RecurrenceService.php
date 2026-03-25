<?php
namespace Modules\WorkOrders\Services;
use Carbon\Carbon;
class RecurrenceService {
    public static function nextOccurrence(string $rrule, Carbon $after): ?Carbon {
        // Minimal RRULE support: FREQ=DAILY|WEEKLY|MONTHLY;INTERVAL=1;BYHOUR=9;BYMINUTE=0
        $parts = [];
        foreach (explode(';', $rrule) as $p) {
            [$k,$v] = array_pad(explode('=', $p, 2), 2, null);
            if ($k) $parts[strtoupper(trim($k))] = strtoupper(trim((string)$v));
        }
        $freq = $parts['FREQ'] ?? 'DAILY';
        $interval = intval($parts['INTERVAL'] ?? 1);
        $h = intval($parts['BYHOUR'] ?? $after->hour);
        $m = intval($parts['BYMINUTE'] ?? $after->minute);
        $candidate = $after->copy();
        if ($freq === 'DAILY') $candidate->addDays($interval);
        elseif ($freq === 'WEEKLY') $candidate->addWeeks($interval);
        elseif ($freq === 'MONTHLY') $candidate->addMonths($interval);
        else $candidate->addDays($interval);
        $candidate->setTime($h, $m);
        return $candidate;
    }
}