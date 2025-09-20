<?php

namespace App\Concerns;

use Illuminate\Support\Facades\Log;
use RuntimeException;

trait HasLog
{
    /**
     * Invalid type log
     *
     * @param  string  $type  // this should be function name or type of varaible
     *
     * @throws RuntimeException
     */
    public static function InvalidTypeLog(string $type, string $value)
    {
        $error = [
            'value'                 => $value,
            "$type used from"       => debug_backtrace()[1]['function'],
            "$type used from class" => debug_backtrace()[1]['class'],
            "$type used in"         => debug_backtrace()[1]['file'],
            "$type used line"       => debug_backtrace()[1]['line'],
        ];

        Log::error('-------------------------------------------------\n');
        Log::error("$type used with invalid value: " . json_encode($error, JSON_THROW_ON_ERROR));
        Log::error('-------------------------------------------------\n');

        throw new RuntimeException('Invalid value: ' . $value . '. Please refresh the page');
    }
}
