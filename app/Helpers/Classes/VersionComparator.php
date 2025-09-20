<?php

namespace App\Helpers\Classes;

use RuntimeException;

class VersionComparator
{
    public static function compareVersion($version1, $version2, $operator = '='): bool
    {
        $normalizedV1 = self::formatToVersion($version1);
        $normalizedV2 = self::formatToVersion($version2);

        $v1Parts = array_map('intval', explode('.', $normalizedV1));

        $v2Parts = array_map('intval', explode('.', $normalizedV2));

        $maxLength = max(count($v1Parts), count($v2Parts));

        $compareResult = 0;

        for ($i = 0; $i < $maxLength; $i++) {
            $v1Part = $v1Parts[$i] ?? 0;
            $v2Part = $v2Parts[$i] ?? 0;

            if ($v1Part > $v2Part) {
                $compareResult = 1;

                break;
            }

            if ($v1Part < $v2Part) {
                $compareResult = -1;

                break;
            }
        }

        return match ($operator) {
            '==', '=' => $compareResult === 0,
            '!=', '<>' => $compareResult !== 0,
            '>'     => $compareResult > 0,
            '>='    => $compareResult >= 0,
            '<'     => $compareResult < 0,
            '<='    => $compareResult <= 0,
            default => throw new RuntimeException("Invalid operator: $operator"),
        };
    }

    public static function formatToVersion($number): string
    {
        $numberStr = (string) $number;

        if (substr_count($numberStr, '.') === 2) {
            return substr($numberStr, 0, 5);
        }

        if (! str_contains($numberStr, '.')) {
            return $number . '.0.0';
        }

        $parts = explode('.', $numberStr);
        $integerPart = $parts[0];
        $decimalPart = $parts[1];

        if ((int) $decimalPart === 0) {
            return $integerPart . '.0.0';
        }

        $decimalPart = str_pad($decimalPart, 2, '0', STR_PAD_RIGHT);

        $firstDigit = $decimalPart[0];
        $secondDigit = $decimalPart[1];

        return $integerPart . '.' . $firstDigit . '.' . $secondDigit;
    }
}
