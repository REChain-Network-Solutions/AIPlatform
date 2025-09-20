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

namespace YooKassa\Model\Receipt;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Class MarkCodeInfo.
 *
 * Код товара (тег в 54 ФЗ — 1163).
 * Обязательный параметр, если одновременно выполняются эти условия:
 * * вы используете Чеки от ЮKassa или онлайн-кассу, обновленную до ФФД 1.2;
 * * товар нужно [маркировать](http://docs.cntd.ru/document/902192509).
 *
 * Должно быть заполнено хотя бы одно поле.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $markCodeRaw Код товара в том виде, в котором он был прочитан сканером (тег в 54 ФЗ — 2000). <br/>Нужно передавать, если используете онлайн-кассу Orange Data. <br/>Пример: ~`010460406000590021N4N57RTCBUZTQ\\u001d2403054002410161218\\u001d1424010191ffd0\\u001g92tIAF/YVpU4roQS3M/m4z78yFq0nc/WsSmLeX6QkF/YVWwy5IMYAeiQ91Xa2m/fFSJcOkb2N+uUUtfr4n0mOX0Q==`
 * @property string $mark_code_raw Код товара в том виде, в котором он был прочитан сканером (тег в 54 ФЗ — 2000). <br/>Нужно передавать, если используете онлайн-кассу Orange Data. <br/>Пример: ~`010460406000590021N4N57RTCBUZTQ\\u001d2403054002410161218\\u001d1424010191ffd0\\u001g92tIAF/YVpU4roQS3M/m4z78yFq0nc/WsSmLeX6QkF/YVWwy5IMYAeiQ91Xa2m/fFSJcOkb2N+uUUtfr4n0mOX0Q==`
 * @property string $unknown Нераспознанный код товара (тег в 54 ФЗ — 1300)
 * @property string $ean_8 Код товара в формате EAN-8 (тег в 54 ФЗ — 1301)
 * @property string $ean_13 Код товара в формате EAN-13 (тег в 54 ФЗ — 1302)
 * @property string $itf_14 Код товара в формате ITF-14 (тег в 54 ФЗ — 1303)
 * @property string $gs_10 Код товара в формате GS1.0 (тег в 54 ФЗ — 1304). <br/>Можно передавать, если используете онлайн-кассу Orange Data, aQsi, Кит Инвест, АТОЛ Онлайн.
 * @property string $gs_1m Код товара в формате GS1.M (тег в 54 ФЗ — 1305)
 * @property string $short Код товара в формате короткого кода маркировки (тег в 54 ФЗ — 1306)
 * @property string $fur Контрольно-идентификационный знак мехового изделия (тег в 54 ФЗ — 1307)
 * @property string $egais_20 Код товара в формате ЕГАИС-2.0 (тег в 54 ФЗ — 1308)
 * @property string $egais_30 Код товара в формате ЕГАИС-3.0 (тег в 54 ФЗ — 1309)
 */
class MarkCodeInfo extends AbstractObject
{
    /** @var int Минимальная длинна поля */
    public const MIN_LENGTH = 1;

    /** @var int Максимальная длина UNKNOWN */
    public const MAX_UNKNOWN_LENGTH = 32;

    /** @var int Длина EAN_8 */
    public const EAN_8_LENGTH = 8;

    /** @var int Длина EAN_13 */
    public const EAN_13_LENGTH = 13;

    /** @var int Длина ITF_14 */
    public const ITF_14_LENGTH = 14;

    /** @var int Максимальная длина GS_10 */
    public const MAX_GS_10_LENGTH = 38;

    /** @var int Максимальная длина GS_1M */
    public const MAX_GS_1M_LENGTH = 200;

    /** @var int Максимальная длина SHORT */
    public const MAX_SHORT_LENGTH = 38;

    /** @var int Длина FUR */
    public const FUR_LENGTH = 20;

    /** @var int Максимальная длина EGAIS_20 */
    public const EGAIS_20_LENGTH = 33;

    /** @var int Длина EGAIS_30 */
    public const EGAIS_30_LENGTH = 14;

    /** @var string|null Код товара в том виде, в котором он был прочитан сканером (тег в 54 ФЗ — 2000). <br/>Нужно передавать, если используете онлайн-кассу Orange Data. <br/>Пример: `010460406000590021N4N57RTCBUZTQ\\u001d2403054002410161218\\u001d1424010191ffd0\\u001g92tIAF/YVpU4roQS3M/m4z78yFq0nc/WsSmLeX6QkF/YVWwy5IMYAeiQ91Xa2m/fFSJcOkb2N+uUUtfr4n0mOX0Q==` */
    #[Assert\Type('string')]
    private ?string $_mark_code_raw = null;

    /** @var string|null Нераспознанный код товара (тег в 54 ФЗ — 1300) */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_UNKNOWN_LENGTH)]
    #[Assert\Length(min: self::MIN_LENGTH)]
    private ?string $_unknown = null;

    /** @var string|null Код товара в формате EAN-8 (тег в 54 ФЗ — 1301) */
    #[Assert\Type('string')]
    #[Assert\Length(exactly: self::EAN_8_LENGTH)]
    private ?string $_ean_8 = null;

    /** @var string|null Код товара в формате EAN-13 (тег в 54 ФЗ — 1302) */
    #[Assert\Type('string')]
    #[Assert\Length(exactly: self::EAN_13_LENGTH)]
    private ?string $_ean_13 = null;

    /** @var string|null Код товара в формате ITF-14 (тег в 54 ФЗ — 1303) */
    #[Assert\Type('string')]
    #[Assert\Length(exactly: self::ITF_14_LENGTH)]
    private ?string $_itf_14 = null;

    /** @var string|null Код товара в формате GS1.0 (тег в 54 ФЗ — 1304). <br/>Можно передавать, если используете онлайн-кассу Orange Data, aQsi, Кит Инвест. */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_GS_10_LENGTH)]
    #[Assert\Length(min: self::MIN_LENGTH)]
    private ?string $_gs_10 = null;

    /** @var string|null Код товара в формате GS1.M (тег в 54 ФЗ — 1305) */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_GS_1M_LENGTH)]
    #[Assert\Length(min: self::MIN_LENGTH)]
    private ?string $_gs_1m = null;

    /** @var string|null Код товара в формате короткого кода маркировки (тег в 54 ФЗ — 1306) */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_SHORT_LENGTH)]
    #[Assert\Length(min: self::MIN_LENGTH)]
    private ?string $_short = null;

    /** @var string|null Контрольно-идентификационный знак мехового изделия (тег в 54 ФЗ — 1307) */
    #[Assert\Type('string')]
    #[Assert\Length(exactly: self::FUR_LENGTH)]
    private ?string $_fur = null;

    /** @var string|null Код товара в формате ЕГАИС-2.0 (тег в 54 ФЗ — 1308) */
    #[Assert\Type('string')]
    #[Assert\Length(exactly: self::EGAIS_20_LENGTH)]
    private ?string $_egais_20 = null;

    /** @var string|null Код товара в формате ЕГАИС-3.0 (тег в 54 ФЗ — 1309) */
    #[Assert\Type('string')]
    #[Assert\Length(exactly: self::EGAIS_30_LENGTH)]
    private ?string $_egais_30 = null;

    /**
     * Возвращает исходный код товара.
     *
     * @return string|null Исходный код товара
     */
    public function getMarkCodeRaw(): ?string
    {
        return $this->_mark_code_raw;
    }

    /**
     * Устанавливает исходный код товара.
     *
     * @param string|null $mark_code_raw Исходный код товара
     * @return self
     */
    public function setMarkCodeRaw(?string $mark_code_raw = null): self
    {
        $this->_mark_code_raw = $this->validatePropertyValue('_mark_code_raw', $mark_code_raw);
        return $this;
    }

    /**
     * Возвращает unknown.
     *
     * @return string|null
     */
    public function getUnknown(): ?string
    {
        return $this->_unknown;
    }

    /**
     * Устанавливает unknown.
     *
     * @param string|null $unknown Нераспознанный код товара (тег в 54 ФЗ — 1300).
     *
     * @return self
     */
    public function setUnknown(?string $unknown = null): self
    {
        $this->_unknown = $this->validatePropertyValue('_unknown', $unknown);
        return $this;
    }

    /**
     * Возвращает ean_8.
     *
     * @return string|null
     */
    public function getEan8(): ?string
    {
        return $this->_ean_8;
    }

    /**
     * Устанавливает ean_8.
     *
     * @param string|null $ean_8 Код товара в формате EAN-8 (тег в 54 ФЗ — 1301).
     *
     * @return self
     */
    public function setEan8(?string $ean_8 = null): self
    {
        $this->_ean_8 = $this->validatePropertyValue('_ean_8', $ean_8);
        return $this;
    }

    /**
     * Возвращает ean_13.
     *
     * @return string|null
     */
    public function getEan13(): ?string
    {
        return $this->_ean_13;
    }

    /**
     * Устанавливает ean_13.
     *
     * @param string|null $ean_13 Код товара в формате EAN-13 (тег в 54 ФЗ — 1302).
     *
     * @return self
     */
    public function setEan13(?string $ean_13 = null): self
    {
        $this->_ean_13 = $this->validatePropertyValue('_ean_13', $ean_13);
        return $this;
    }

    /**
     * Возвращает itf_14.
     *
     * @return string|null
     */
    public function getItf14(): ?string
    {
        return $this->_itf_14;
    }

    /**
     * Устанавливает itf_14.
     *
     * @param string|null $itf_14 Код товара в формате ITF-14 (тег в 54 ФЗ — 1303).
     *
     * @return self
     */
    public function setItf14(?string $itf_14 = null): self
    {
        $this->_itf_14 = $this->validatePropertyValue('_itf_14', $itf_14);
        return $this;
    }

    /**
     * Возвращает gs_10.
     *
     * @return string|null
     */
    public function getGs10(): ?string
    {
        return $this->_gs_10;
    }

    /**
     * Устанавливает gs_10.
     *
     * @param string|null $gs_10 Код товара в формате GS1.0 (тег в 54 ФЗ — 1304). <br/>Онлайн-кассы, которые поддерживают этот параметр: **Orange Data**, **aQsi**, **Кит Инвест**.
     *
     * @return self
     */
    public function setGs10(?string $gs_10 = null): self
    {
        $this->_gs_10 = $this->validatePropertyValue('_gs_10', $gs_10);
        return $this;
    }

    /**
     * Возвращает gs_1m.
     *
     * @return string|null
     */
    public function getGs1m(): ?string
    {
        return $this->_gs_1m;
    }

    /**
     * Устанавливает gs_1m.
     *
     * @param string|null $gs_1m Код товара в формате GS1.M (тег в 54 ФЗ — 1305).
     *
     * @return self
     */
    public function setGs1m(?string $gs_1m = null): self
    {
        $this->_gs_1m = $this->validatePropertyValue('_gs_1m', $gs_1m);
        return $this;
    }

    /**
     * Возвращает short.
     *
     * @return string|null
     */
    public function getShort(): ?string
    {
        return $this->_short;
    }

    /**
     * Устанавливает short.
     *
     * @param string|null $short Код товара в формате короткого кода маркировки (тег в 54 ФЗ — 1306).
     *
     * @return self
     */
    public function setShort(?string $short = null): self
    {
        $this->_short = $this->validatePropertyValue('_short', $short);
        return $this;
    }

    /**
     * Возвращает fur.
     *
     * @return string|null
     */
    public function getFur(): ?string
    {
        return $this->_fur;
    }

    /**
     * Устанавливает fur.
     *
     * @param string|null $fur Контрольно-идентификационный знак мехового изделия (тег в 54 ФЗ — 1307).
     *
     * @return self
     */
    public function setFur(?string $fur = null): self
    {
        $this->_fur = $this->validatePropertyValue('_fur', $fur);
        return $this;
    }

    /**
     * Возвращает egais_20.
     *
     * @return string|null
     */
    public function getEgais20(): ?string
    {
        return $this->_egais_20;
    }

    /**
     * Устанавливает egais_20.
     *
     * @param string|null $egais_20 Код товара в формате ЕГАИС-2.0 (тег в 54 ФЗ — 1308).
     *
     * @return self
     */
    public function setEgais20(?string $egais_20 = null): self
    {
        $this->_egais_20 = $this->validatePropertyValue('_egais_20', $egais_20);
        return $this;
    }

    /**
     * Возвращает egais_30.
     *
     * @return string|null
     */
    public function getEgais30(): ?string
    {
        return $this->_egais_30;
    }

    /**
     * Устанавливает egais_30.
     *
     * @param string|null $egais_30 Код товара в формате ЕГАИС-3.0 (тег в 54 ФЗ — 1309).
     *
     * @return self
     */
    public function setEgais30(?string $egais_30 = null): self
    {
        $this->_egais_30 = $this->validatePropertyValue('_egais_30', $egais_30);
        return $this;
    }

    /**
     * {@inheritdoc}
     */
    public function jsonSerialize(): array
    {
        $fields = [
            'mark_code_raw', 'unknown', 'ean_8', 'ean_13', 'itf_14', 'gs_10', 'gs_1m', 'short', 'fur', 'egais_20', 'egais_30',
        ];
        $result = [];
        foreach ($fields as $key) {
            $value = $this->{$key};
            if (null !== $value) {
                $result[$key] = $value;
            }
        }

        return $result;
    }
}
