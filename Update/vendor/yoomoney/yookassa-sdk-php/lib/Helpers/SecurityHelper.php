<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Helpers;

use Exception;
use RuntimeException;

/**
 * Класс, представляющий модель SecurityHelper.
 *
 * Класс для проверки IP адреса входящих запросов от API кассы.
 *
 * @category Class
 * @package  YooKassa\Helpers
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class SecurityHelper
{
    /**
     * Проверяет формат IP адреса и вызывает соответствующие методы для проверки среди IPv4 и IPv6 адресов Юkassa.
     *
     * @param mixed $ip - IPv4 или IPv6 адрес webhook уведомления
     *
     * @throws Exception - исключение будет выброшено, если не удастся установить формат IP адреса
     */
    public function isIPTrusted(mixed $ip): bool
    {
        if (!$this->isIPv6($ip)) {
            return $this->checkInIPv4TrustedList($ip);
        }

        if (!$this->isIPv4($ip)) {
            return $this->checkInIPv6TrustedList($ip);
        }

        throw new RuntimeException(
            'Could not recognize IPv4 or IPv6: ' . $ip
        );
    }

    /**
     * Проверяет, является ли переданное в функцию значение IPv6 адресом
     *
     * @param mixed $ip - IP адрес
     *
     * @return bool - true - является, false - не является
     */
    private function isIPv6(mixed $ip): bool
    {
        return false !== filter_var($ip, FILTER_VALIDATE_IP, FILTER_FLAG_IPV6);
    }

    /**
     * Проверяет, является ли переданное в функцию значение IPv4 адресом
     *
     * @param mixed $ip - IP адрес
     *
     * @return bool - true - является, false - не является
     */
    private function isIPv4(mixed $ip): bool
    {
        return false !== filter_var($ip, FILTER_VALIDATE_IP, FILTER_FLAG_IPV4);
    }

    /**
     * Проверяет IPv4 адрес в списке IPv4 адресов Юkassa.
     *
     * @param mixed $ip - IPv4 адрес
     */
    private function checkInIPv4TrustedList(mixed $ip): bool
    {
        foreach ($this->getIPv4TrustedList() as $range) {
            if ($this->isIPInV4Range($ip, $range)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Проверяет IPv6 адрес в списке IPv6 адресов Юkassa.
     *
     * @param mixed $ip - IPv6 адрес
     */
    private function checkInIPv6TrustedList(mixed $ip): bool
    {
        foreach ($this->getIPv6TrustedList() as $range) {
            if ($this->isIPInV6Range($ip, $range)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Осуществляет проверку, входит ли IPv4 адрес $ip в диапазон IPv4 адресов $range.
     *
     * @param mixed $ip - IPv4 адрес
     * @param mixed $range - IPv4 адрес, или диапазон IPv4 адресов в формате CIDR
     */
    private function isIPInV4Range(mixed $ip, mixed $range): bool
    {
        $ip_dec = ip2long($ip);

        if (!str_contains($range, '/')) {
            return ip2long($ip) === ip2long($range);
        }
        [$range, $netmask] = explode('/', $range, 2);
        [$a, $b, $c, $d] = explode('.', $range);

        $range = sprintf('%u.%u.%u.%u', $a, $b, $c, $d);
        $range_dec = ip2long($range);

        $wildcard_dec = 2 ** (32 - (int) $netmask) - 1;
        $netmask_dec = ~$wildcard_dec;

        return ($ip_dec & $netmask_dec) === ($range_dec & $netmask_dec);
    }

    /**
     * Осуществляет проверку, входит ли IPv6 адрес $ip в диапазон IPv6 адресов $range.
     *
     * @param mixed $ip
     * @param array $range
     * @return bool
     */
    private function isIPInV6Range(mixed $ip, array $range): bool
    {
        $firstInRange = inet_pton($range[0]);
        $lastInRange = inet_pton($range[1]);

        $ip = inet_pton($ip);

        return (strlen($ip) === strlen($firstInRange))
                && ($ip >= $firstInRange && $ip <= $lastInRange);
    }

    /**
     * Возвращает список диапазонов IPv4 адресов в формате CIDR и отдельных IPv4 адресов
     * с которых Юkassa может отправлять уведомления.
     *
     * @return string[]
     */
    private function getIPv4TrustedList(): array
    {
        return [
            '185.71.76.0/27',
            '185.71.77.0/27',
            '77.75.153.0/25',
            '77.75.154.128/25',
            '77.75.156.11',
            '77.75.156.35',
        ];
    }

    /**
     * Возвращает список диапазонов IPv6 адресов с которых Юkassa может отправлять уведомления.
     *
     * @return string[][]
     */
    private function getIPv6TrustedList(): array
    {
        return [
            [
                '2a02:5180:0000:1509:0000:0000:0000:0000',
                '2a02:5180:0000:1509:ffff:ffff:ffff:ffff',
            ],
            [
                '2a02:5180:0000:2655:0000:0000:0000:0000',
                '2a02:5180:0000:2655:ffff:ffff:ffff:ffff',
            ],
            [
                '2a02:5180:0000:1533:0000:0000:0000:0000',
                '2a02:5180:0000:1533:ffff:ffff:ffff:ffff',
            ],
            [
                '2a02:5180:0000:2669:0000:0000:0000:0000',
                '2a02:5180:0000:2669:ffff:ffff:ffff:ffff',
            ],
        ];
    }
}
