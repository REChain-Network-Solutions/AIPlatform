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

use YooKassa\Model\Receipt\MarkCodeInfo;

/**
 * Класс, представляющий модель ProductCode.
 *
 * Класс для формирования тега 1162 на основе кода в формате Data Matrix.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @example 04-product-code.php 6 7 Вариант через метод
 * @example 04-product-code.php 15 7 Вариант через массив
 *
 * @see https://git.yoomoney.ru/projects/SDK/repos/yookassa-sdk-php/browse/lib/Helpers/ProductCode.php
 */
class ProductCode
{
    /** @var string Код типа маркировки DataMatrix */
    public const PREFIX_DATA_MATRIX = '444D';

    /** @var string Код типа маркировки UNKNOWN */
    public const PREFIX_UNKNOWN = '0000';

    /** @var string Код типа маркировки EAN_8 */
    public const PREFIX_EAN_8 = '4508';

    /** @var string Код типа маркировки EAN_13 */
    public const PREFIX_EAN_13 = '450D';

    /** @var string Код типа маркировки ITF_14 */
    public const PREFIX_ITF_14 = '4909';

    /** @var string Код типа маркировки FUR */
    public const PREFIX_FUR = '5246';

    /** @var string Код типа маркировки EGAIS_20 */
    public const PREFIX_EGAIS_20 = 'C514';

    /** @var string Код типа маркировки EGAIS_30 */
    public const PREFIX_EGAIS_30 = 'C51E';

    /** @var string Тип маркировки UNKNOWN */
    public const TYPE_UNKNOWN = 'unknown';

    /** @var string Тип маркировки EAN_8 */
    public const TYPE_EAN_8 = 'ean_8';

    /** @var string Тип маркировки EAN_13 */
    public const TYPE_EAN_13 = 'ean_13';

    /** @var string Тип маркировки ITF_14 */
    public const TYPE_ITF_14 = 'itf_14';

    /** @var string Тип маркировки GS_10 */
    public const TYPE_GS_10 = 'gs_10';

    /** @var string Тип маркировки GS_1M */
    public const TYPE_GS_1M = 'gs_1m';

    /** @var string Тип маркировки SHORT */
    public const TYPE_SHORT = 'short';

    /** @var string Тип маркировки FUR */
    public const TYPE_FUR = 'fur';

    /** @var string Тип маркировки EGAIS_20 */
    public const TYPE_EGAIS_20 = 'egais_20';

    /** @var string Тип маркировки EGAIS_30 */
    public const TYPE_EGAIS_30 = 'egais_30';

    /** @var string Идентификатор применения (идентификационный номер единицы товара) */
    public const AI_GTIN = '01';

    /** @var string Идентификатор применения (серийный номер) */
    public const AI_SERIAL = '21';

    /** @var string Дополнительный идентификатор применения (цена единицы измерения товара) */
    public const AI_SUM = '8005';

    /** @var int Максимальная длина последовательности для кода продукта unknown */
    public const MAX_PRODUCT_CODE_LENGTH = 30;

    /** @var int Максимальная длина последовательности для кода маркировки типа unknown */
    public const MAX_MARK_CODE_LENGTH = 32;

    /** @var string|null Код типа маркировки */
    private ?string $prefix = null;

    /**
     * @var string|null Тип маркировки
     *
     * @example unknown
     */
    private ?string $type = null;

    /**
     * @var string|null Global Trade Item Number
     *             Глобальный номер товарной продукции в единой международной базе товаров GS1 https://ru.wikipedia.org/wiki/GS1
     *
     * @example 04630037591316
     */
    private ?string $gtin = null;

    /**
     * @var string|null Серийный номер товара
     *
     * @example sgEKKPPcS25y5
     */
    private ?string $serial = null;

    /**
     * @var null|array Массив дополнительных идентификаторов применения
     */
    private ?array $appIdentifiers = null;

    /**
     * @var string|null Сформированный тег 1162. Формат: hex([prefix]+gtin+serial)
     *
     * @example 04 36 03 BE F5 14 73  67  45  4b  4b  50  50  63  53  32  35  79  35
     */
    private ?string $result = null;

    /**
     * @var MarkCodeInfo|null Сформированный код товара (тег в 54 ФЗ — 1163)
     */
    private ?MarkCodeInfo $markCodeInfo = null;

    /** @var bool Флаг использования кода типа маркировки */
    private bool $usePrefix = false;

    /**
     * ProductCode constructor.
     *
     * @param null|string $codeDataMatrix Строка, расшифрованная из QR-кода
     * @param bool|string $usePrefix Нужен ли код типа маркировки в результате
     */
    public function __construct(?string $codeDataMatrix = null, mixed $usePrefix = true)
    {
        $this->preparePrefix($usePrefix);

        if (!empty($codeDataMatrix) && $this->parseCodeMatrixData($codeDataMatrix)) {
            $this->result = $this->calcResult();
        }
    }

    /**
     * Приводит объект к строке.
     */
    public function __toString(): string
    {
        return $this->getResult();
    }

    /**
     * Возвращает код типа маркировки.
     *
     * @return ?string Код типа маркировки
     */
    public function getPrefix(): ?string
    {
        return $this->prefix;
    }

    /**
     * Устанавливает код типа маркировки.
     *
     * @param int|string|null $prefix Код типа маркировки
     */
    public function setPrefix(mixed $prefix): ProductCode
    {
        if (null === $prefix || '' === $prefix) {
            $this->prefix = null;

            return $this;
        }

        if (is_int($prefix)) {
            $prefix = dechex($prefix);
        }
        $this->prefix = str_pad($prefix, 4, '0', STR_PAD_LEFT);

        return $this;
    }

    /**
     * Возвращает тип маркировки.
     *
     * @return string|null Тип маркировки
     */
    public function getType(): ?string
    {
        return $this->type;
    }

    /**
     * Устанавливает тип маркировки.
     *
     * @param string $type Тип маркировки
     */
    public function setType(string $type): void
    {
        $this->type = $type;
    }

    /**
     * Возвращает глобальный номер товарной продукции.
     *
     * @return string|null Глобальный номер товарной продукции
     */
    public function getGtin(): ?string
    {
        return $this->gtin;
    }

    /**
     * Устанавливает глобальный номер товарной продукции.
     *
     * @param string|null $gtin Глобальный номер товарной продукции
     * @return ProductCode
     */
    public function setGtin(?string $gtin): ProductCode
    {
        if (null === $gtin || '' === $gtin) {
            $this->gtin = null;
        } else {
            $this->gtin = $gtin;
        }

        return $this;
    }

    /**
     * Возвращает серийный номер товара.
     *
     * @return string|null Серийный номер товара
     */
    public function getSerial(): ?string
    {
        return $this->serial;
    }

    /**
     * Устанавливает серийный номер товара.
     *
     * @param string|null $serial Серийный номер товара
     * @return ProductCode
     */
    public function setSerial(?string $serial): ProductCode
    {
        if (null === $serial || '' === $serial) {
            $this->prefix = null;
        } else {
            $this->serial = $serial;
        }

        return $this;
    }

    /**
     * Возвращает массив дополнительных идентификаторов применения.
     *
     * @return null|array Массив дополнительных идентификаторов применения
     */
    public function getAppIdentifiers(): ?array
    {
        return $this->appIdentifiers;
    }

    /**
     * Устанавливает массив дополнительных идентификаторов применения.
     *
     * @param null|array $appIdentifiers Массив дополнительных идентификаторов применения
     */
    public function setAppIdentifiers(?array $appIdentifiers): void
    {
        $this->appIdentifiers = $appIdentifiers;
    }

    /**
     * Возвращает сформированный тег 1162.
     *
     * @return string Сформированный тег 1162
     */
    public function getResult(): string
    {
        if (!$this->result) {
            $this->result = $this->calcResult();
        }

        return $this->result;
    }

    public function getMarkCodeInfo(): ?MarkCodeInfo
    {
        return $this->markCodeInfo;
    }

    /**
     * @param array|MarkCodeInfo|string $markCodeInfo
     */
    public function setMarkCodeInfo(mixed $markCodeInfo): void
    {
        if (is_array($markCodeInfo)) {
            $markCodeInfo = new MarkCodeInfo($markCodeInfo);
        }
        if (is_string($markCodeInfo)) {
            $markCodeInfo = new MarkCodeInfo([
                $this->getType() => $markCodeInfo,
            ]);
        }

        $this->markCodeInfo = $markCodeInfo;
    }

    /**
     * Возвращает флаг использования кода типа маркировки.
     */
    public function isUsePrefix(): bool
    {
        return $this->usePrefix;
    }

    /**
     * Устанавливает флаг использования кода типа маркировки.
     *
     * @param bool $usePrefix Флаг использования кода типа маркировки
     * @return ProductCode
     */
    public function setUsePrefix(bool $usePrefix): ProductCode
    {
        $this->usePrefix = $usePrefix;

        return $this;
    }

    /**
     * Формирует тег 1162.
     *
     * @return string Сформированный тег 1162
     */
    public function calcResult(): string
    {
        $result = '';

        if (!$this->validate()) {
            return $result;
        }

        if ($this->isUsePrefix()) {
            $result = $this->getPrefix() ?: self::PREFIX_DATA_MATRIX;
        }

        switch ($this->getType()) {
            case self::TYPE_EAN_8:
            case self::TYPE_EAN_13:
            case self::TYPE_ITF_14:
                $result .= $this->numToHex($this->getGtin());

                break;

            case self::TYPE_FUR:
            case self::TYPE_EGAIS_20:
            case self::TYPE_EGAIS_30:
            case self::TYPE_UNKNOWN:
                $result .= $this->strToHex($this->getGtin());

                break;

            case self::TYPE_SHORT:
                $result .= $this->numToHex($this->getGtin());
                $result .= $this->strToHex($this->getSerial());

                break;

            case self::TYPE_GS_1M:
            case self::TYPE_GS_10:
                $result .= $this->numToHex($this->getGtin());
                $result .= $this->strToHex($this->getSerial());
                if ($sum = $this->getAIValue(self::AI_SUM)) {
                    $result .= $this->strToHex($sum);
                }

                break;
        }

        return $this->chunkStr($result);
    }

    /**
     * Проверяет заполненность необходимых свойств.
     */
    public function validate(): bool
    {
        if (!$this->getType()) {
            return false;
        }

        return match ($this->getType()) {
            self::TYPE_EAN_8, self::TYPE_EAN_13, self::TYPE_ITF_14, self::TYPE_FUR, self::TYPE_EGAIS_20, self::TYPE_EGAIS_30, self::TYPE_UNKNOWN => null !== $this->getGtin(),
            self::TYPE_GS_10, self::TYPE_GS_1M, self::TYPE_SHORT => $this->getGtin() && $this->getSerial(),
            default => false,
        };

    }

    /**
     * Устанавливает prefix и usePrefix в зависимости от входящего параметра.
     *
     * @param mixed $usePrefix Код типа маркировки или bool
     */
    private function preparePrefix(mixed $usePrefix): void
    {
        if ($usePrefix) {
            $this->setUsePrefix(true);
            if (is_string($usePrefix) || is_int($usePrefix)) {
                $this->setPrefix($usePrefix);
            } else {
                $this->setPrefix(self::PREFIX_UNKNOWN);
            }
        } else {
            $this->setUsePrefix(false);
            $this->setPrefix(null);
        }
    }

    /**
     * Извлекает необходимые данные из строки, расшифрованной из QR-кода и устанавливает соответствующие свойства.
     * Возвращает результат в виде bool.
     *
     * @param string $codeDataMatrix Строки, расшифрованная из QR-кода
     *
     * @return false
     */
    private function parseCodeMatrixData(string $codeDataMatrix): bool
    {
        $this->fillData(
            $this->parseScanString($codeDataMatrix)
                ?: ['type' => self::TYPE_UNKNOWN, 'code' => $codeDataMatrix]
        );

        return $this->validate();
    }

    /**
     * Заполняет поля объекта из массива данных.
     *
     * @param array $data Массив данных
     */
    private function fillData(array $data): void
    {
        $this->setType($data['type']);
        $this->setPrefix($this->getPrefixByType($data['type']));

        switch ($this->getType()) {
            case self::TYPE_EAN_8:
            case self::TYPE_EAN_13:
            case self::TYPE_ITF_14:
            case self::TYPE_FUR:
            case self::TYPE_EGAIS_30:
                $this->setGtin($data['match1']);
                $this->setMarkCodeInfo($this->getGtin());

                break;

            case self::TYPE_EGAIS_20:
                $this->setGtin($data['match2']);
                $this->setMarkCodeInfo($this->getGtin());

                break;

            case self::TYPE_SHORT:
                $this->setGtin($data['match1']);
                $this->setSerial($data['match2']);
                $this->setMarkCodeInfo(self::AI_GTIN . $this->getGtin() . self::AI_SERIAL . $this->getSerial());

                break;

            case self::TYPE_GS_1M:
            case self::TYPE_GS_10:
                $this->setGtin($data['match1']);
                if (!empty($data['split']) && count($data['split']) > 1) {
                    $this->setSerial(array_shift($data['split']));
                    $this->setAppIdentifiers($data['split']);
                } else {
                    $this->setSerial($data['match2']);
                }
                $this->setMarkCodeInfo(self::AI_GTIN . $this->getGtin() . self::AI_SERIAL . $this->getSerial());

                break;

            case self::TYPE_UNKNOWN:
                $this->setGtin(strlen($data['code']) > self::MAX_PRODUCT_CODE_LENGTH ? substr($data['code'], 0, self::MAX_PRODUCT_CODE_LENGTH) : $data['code']);
                $this->setSerial(strlen($data['code']) > self::MAX_MARK_CODE_LENGTH ? substr($data['code'], 0, self::MAX_MARK_CODE_LENGTH) : $data['code']);
                $this->setMarkCodeInfo($this->getSerial());

                break;
        }
    }

    /**
     * Возвращает массив данных, полученных по правилам из считанной сканером последовательности.
     *
     * @param string $code Считанная сканером последовательность
     *
     * @return array|void Массив данных, полученных по правилам
     */
    private function parseScanString(string $code)
    {
        foreach ($this->getMarkCodeRules() as $codeType => $rule) {
            if (!empty($rule['length']) && strlen($code) !== $rule['length']) {
                continue;
            }
            preg_match($rule['pattern'], $code, $matches);
            if ($rule['matches'][1] && empty($matches[1])) {
                continue;
            }
            if ($rule['matches'][2] && empty($matches[2])) {
                continue;
            }
            if (!empty($rule['split'])) {
                $split = preg_split($rule['split'], $matches[2]);
            }

            return [
                'type' => $codeType,
                'code' => $code,
                'rules' => $rule['matches'],
                'match1' => $rule['matches'][1] ? $matches[1] : null,
                'match2' => $rule['matches'][2] ? $matches[2] : null,
                'split' => !empty($split) ? $split : null,
            ];
        }
    }

    /**
     * Возвращает список правил определения типа маркировки.
     *
     * @return array[]
     */
    private function getMarkCodeRules(): array
    {
        return [
            self::TYPE_GS_1M => [
                'pattern' => '#^01(\d{14})21(.+)((91(.+)92(.+))|(93[\w!"%&\'()*+,-./_:;=<>?]{4}(.*)))$#ui',
                'matches' => [1 => true, 2 => true],
                'split' => '#\\\u001d|\x{001d}#ui',
            ],
            self::TYPE_GS_10 => [
                'pattern' => '#^01(\d{14})21(.+)$#ui',
                'matches' => [1 => true, 2 => true],
                'split' => '#\\\u001d|\x{001d}#ui',
            ],
            self::TYPE_SHORT => [
                'length' => 29,
                'pattern' => '#^(\d{14})(.+)$#i',
                'matches' => [1 => true, 2 => true],
            ],
            self::TYPE_EGAIS_20 => [
                'length' => 68,
                'pattern' => '#^(.{8})(.{33})(.+)$#ui',
                'matches' => [1 => false, 2 => true],
            ],
            self::TYPE_EGAIS_30 => [
                'length' => 150,
                'pattern' => '#^(.{14})(.+)$#ui',
                'matches' => [1 => true, 2 => false],
            ],
            self::TYPE_ITF_14 => [
                'length' => 14,
                'pattern' => '#^(0\d{13})$#ui',
                'matches' => [1 => true, 2 => false],
            ],
            self::TYPE_EAN_13 => [
                'length' => 13,
                'pattern' => '#^(\d{13})$#ui',
                'matches' => [1 => true, 2 => false],
            ],
            self::TYPE_EAN_8 => [
                'length' => 8,
                'pattern' => '#^(\d{8})$#ui',
                'matches' => [1 => true, 2 => false],
            ],
            self::TYPE_FUR => [
                'length' => 20,
                'pattern' => '#^((\w{2})-(\d{6})-(\w{10}))$#ui',
                'matches' => [1 => true, 2 => false],
            ],
        ];
    }

    /**
     * Возвращает значение идентификатора применения, если он присутствует
     *
     * @param string $appIdentifier Идентификатор применения
     *
     * @return null|string Значение идентификатора применения
     */
    private function getAIValue(string $appIdentifier): ?string
    {
        if (!$this->getAppIdentifiers()) {
            return null;
        }
        foreach ($this->getAppIdentifiers() as $item) {
            if (str_starts_with($item, $appIdentifier)) {
                return str_replace($appIdentifier, '', $item);
            }
        }

        return null;
    }

    /**
     * Возвращает префикс кода товара типу маркировки.
     *
     * @param null|string $type Тип маркировки
     *
     * @return null|string Префикс кода товара
     */
    private function getPrefixByType(?string $type = null): ?string
    {
        if (!$type) {
            $type = $this->getType();
        }
        $map = [
            self::TYPE_UNKNOWN => self::PREFIX_UNKNOWN,
            self::TYPE_EAN_8 => self::PREFIX_EAN_8,
            self::TYPE_EAN_13 => self::PREFIX_EAN_13,
            self::TYPE_ITF_14 => self::PREFIX_ITF_14,
            self::TYPE_GS_10 => self::PREFIX_DATA_MATRIX,
            self::TYPE_GS_1M => self::PREFIX_DATA_MATRIX,
            self::TYPE_SHORT => self::PREFIX_DATA_MATRIX,
            self::TYPE_FUR => self::PREFIX_FUR,
            self::TYPE_EGAIS_20 => self::PREFIX_EGAIS_20,
            self::TYPE_EGAIS_30 => self::PREFIX_EGAIS_30,
        ];

        return !empty($map[$type]) ? $map[$type] : self::PREFIX_UNKNOWN;
    }

    /**
     * Разбивает пробелами строку на пары символов и переводит в верхний регистр
     *
     * @param string $string Подготовленная к разбиению строка
     */
    private function chunkStr(string $string): string
    {
        return strtoupper(trim(chunk_split($string, 2, ' ')));
    }

    /**
     * Переводит десятичное число в шестнадцатеричный вид и дополняет нулями до 12 символов слева.
     *
     * @param string $string Входящее число (Глобальный номер товарной продукции)
     */
    private function numToHex(string $string): string
    {
        return str_pad($this->baseConvert($string), 12, '0', STR_PAD_LEFT);
    }

    /**
     * Переводит число из одной системы исчисления в другую
     * Замена dechex() для 32-битных версии PHP.
     */
    private function baseConvert(string $numString, int $fromBase = 10, int $toBase = 16): string
    {
        $chars = '0123456789abcdefghijklmnopqrstuvwxyz';
        $toString = substr($chars, 0, $toBase);

        $length = mb_strlen($numString);
        $result = '';
        $number = [];

        for ($i = 0; $i < $length; $i++) {
            $number[$i] = strpos($chars, $numString[$i]);
        }

        do {
            $divide = 0;
            $newLen = 0;
            for ($i = 0; $i < $length; $i++) {
                $divide = $divide * $fromBase + $number[$i];
                if ($divide >= $toBase) {
                    $number[$newLen++] = (int) ($divide / $toBase);
                    $divide %= $toBase;
                } elseif ($newLen > 0) {
                    $number[$newLen++] = 0;
                }
            }
            $length = $newLen;
            $result = $toString[$divide] . $result;
        } while (0 !== $newLen);

        return $result;
    }

    /**
     * Переводит строку в шестнадцатеричный вид.
     *
     * @param string $string Входящая строка (Серийный номер товара)
     */
    private function strToHex(string $string): string
    {
        $hex = '';
        $length = mb_strlen($string);
        for ($i = 0; $i < $length; $i++) {
            $ord = ord($string[$i]);
            $hexCode = dechex($ord);
            $hex .= substr('0' . $hexCode, -2);
        }

        return $hex;
    }

    /**
     * Переводит строку из шестнадцатеричного вида в обычный
     * Нужен для тестирования.
     *
     * @param string $hex Входящая строка в шестнадцатеричном виде
     */
    private function hexToStr(string $hex): string
    {
        $string = '';
        for ($i = 0; $i < strlen($hex) - 1; $i += 2) {
            $string .= chr(hexdec($hex[$i] . $hex[$i + 1]));
        }

        return $string;
    }
}
