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

namespace YooKassa\Client;

use YooKassa\Client;

/**
 * Класс, представляющий модель UserAgent.
 *
 * Класс для создания заголовка User-Agent в запросах к API.
 *
 * @category Class
 * @package  YooKassa
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class UserAgent
{
    /** Имя заголовка для User-Agent */
    public const HEADER = 'YM-User-Agent';

    /** Разделитель части заголовка и её версии */
    public const VERSION_DELIMITER = '/';

    /** Разделитель между частями заголовка */
    public const PART_DELIMITER = ' ';

    /** @var null|string Версия операционной системы */
    private ?string $_os = null;

    /** @var null|string Версия PHP */
    private ?string $_php = null;

    /** @var null|string Версия фреймворка */
    private ?string $_framework = null;

    /** @var null|string Версия CMS */
    private ?string $_cms = null;

    /** @var null|string Версия модуля */
    private ?string $_module = null;

    /** @var null|string Версия SDK */
    private ?string $_sdk = null;

    /**
     * Конструктор UserAgent.
     */
    public function __construct()
    {
        if ($os = $this->defineOs()) {
            $this->setOs($os['name'], $os['version']);
        }
        if ($php = $this->definePhp()) {
            $this->setPhp($php['name'], $php['version']);
        }
        $this->setSdk('YooKassa.PHP', Client::SDK_VERSION);
    }

    /**
     * Формирует конечную строку из составных частей.
     */
    public function getHeaderString(): string
    {
        $result = [];

        $result[] = $this->getOs();
        $result[] = $this->getPhp();

        if ($string = $this->getFramework()) {
            $result[] = $string;
        }
        if ($string = $this->getCms()) {
            $result[] = $string;
        }
        if ($string = $this->getModule()) {
            $result[] = $string;
        }

        $result[] = $this->getSdk();

        return implode(self::PART_DELIMITER, $result);
    }

    /**
     * Возвращает версию операционной системы.
     */
    public function getOs(): ?string
    {
        return $this->_os;
    }

    /**
     * Возвращает версию PHP.
     */
    public function getPhp(): ?string
    {
        return $this->_php;
    }

    /**
     * Возвращает версию фреймворка.
     */
    public function getFramework(): ?string
    {
        return $this->_framework;
    }

    /**
     * Устанавливает версию фреймворка.
     */
    public function setFramework(string $name, string $version): void
    {
        $this->_framework = $this->createVersion($name, $version);
    }

    /**
     * Возвращает версию CMS.
     */
    public function getCms(): ?string
    {
        return $this->_cms;
    }

    /**
     * Устанавливает версию CMS.
     */
    public function setCms(string $name, string $version): void
    {
        $this->_cms = $this->createVersion($name, $version);
    }

    /**
     * Возвращает версию модуля.
     */
    public function getModule(): ?string
    {
        return $this->_module;
    }

    /**
     * Устанавливает версию модуля.
     */
    public function setModule(string $name, string $version): void
    {
        $this->_module = $this->createVersion($name, $version);
    }

    /**
     * Возвращает версию SDK.
     */
    public function getSdk(): ?string
    {
        return $this->_sdk;
    }

    /**
     * Создание строки версии компонента.
     */
    public function createVersion(string $name, string $version): string
    {
        return str_replace([self::PART_DELIMITER, self::VERSION_DELIMITER], '.', trim($name))
             . self::VERSION_DELIMITER
             . str_replace([self::PART_DELIMITER, self::VERSION_DELIMITER], '.', trim($version));
    }

    /**
     * Устанавливает версию операционной системы.
     */
    private function setOs(string $name, string $version): void
    {
        $this->_os = $this->createVersion($name, $version);
    }

    /**
     * Устанавливает версию PHP.
     */
    private function setPhp(string $name, string $version): void
    {
        $this->_php = $this->createVersion($name, $version);
    }

    /**
     * Устанавливает версию SDK.
     */
    private function setSdk(string $name, string $version): void
    {
        $this->_sdk = $this->createVersion($name, $version);
    }

    /**
     * Попытка определить систему.
     */
    private function defineOs(): array
    {
        if (PHP_OS_FAMILY === 'Linux') {
            if ($result = $this->parseSimpleLinuxRelease()) {
                return $result;
            }
            if ($result = $this->parseSmartLinuxRelease()) {
                return $result;
            }
        } else {
            return ['name' => php_uname('s'), 'version' => php_uname('r')];
        }

        return ['name' => 'Undefined', 'version' => '0.0.0'];
    }

    /**
     * Возвращает информацию о версии системы
     * Используется сложный вариант
     */
    private function parseSmartLinuxRelease(): ?array
    {
        $vars = [];

        if ($files = glob('/etc/*elease')) {
            foreach ($files as $file) {
                if (is_file($file)) {
                    $lines = array_filter(array_map([$this, 'callbackSmartLinux'], file($file)));
                    if (is_array($lines)) {
                        foreach ($lines as $line) {
                            $vars[strtoupper($line[0])] = trim($line[1]);
                        }
                    }
                }
            }

            if (!empty($vars['NAME']) && !empty($vars['VERSION_ID'])) {
                return ['name' => $vars['NAME'], 'version' => $vars['VERSION_ID']];
            }
        }

        return null;
    }

    /**
     * @param string $line
     * @return array|bool
     */
    private static function callbackSmartLinux(string $line): bool|array
    {
        $parts = explode('=', $line);
        if (2 !== count($parts)) {
            return false;
        }
        $parts[1] = trim(str_replace(['"', "'"], '', $parts[1]));

        return $parts;
    }

    /**
     * Возвращает информацию о версии системы.
     *
     * Используется простой вариант
     */
    private function parseSimpleLinuxRelease(): ?array
    {
        $vars = [];

        if ($files = glob('/etc/*elease')) {
            foreach ($files as $file) {
                if (is_file($file)) {
                    $data = array_map([$this, 'callbackSimpleLinux'], file($file));
                    if (!empty($data)) {
                        $array = array_shift($data);
                        if (!empty($array) && is_array($array)) {
                            $vars = array_merge($vars, $array);
                        }
                    }
                }
            }
        }

        return !empty($vars['name']) && !empty($vars['version']) ? $vars : null;
    }

    private static function callbackSimpleLinux(string $line): array
    {
        $parse = [];
        preg_match('/(.+) release (.+) (.+)/iu', $line, $parts);
        if (!empty($parts[1])) {
            $parse['name'] = str_replace(' ', '.', trim($parts[1]));
        }
        if (!empty($parts[2])) {
            $parse['version'] = trim($parts[2]);
        }

        return $parse;
    }

    /**
     * Определение версии PHP.
     */
    private function definePhp(): array
    {
        return [
            'name' => 'PHP',
            'version' => PHP_MAJOR_VERSION . '.' . PHP_MINOR_VERSION . '.' . PHP_RELEASE_VERSION,
        ];
    }
}
