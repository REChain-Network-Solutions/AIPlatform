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

namespace YooKassa\Model\Settings;

use YooKassa\Common\AbstractEnum;

/**
 * Класс, представляющий модель FiscalizationProvider.
 *
 * Решение ЮKassa, которое магазин использует для отправки чеков.
 * Возможные значения:
 * - [Чеки для самозанятых](https://yookassa.ru/developers/payment-acceptance/receipts/self-employed/basics) — `fns`
 * - [54-ФЗ: Чеки от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — `avanpost`
 * - [54-ФЗ: сторонняя онлайн-касса](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) (наименование онлайн-кассы) — ~`a_qsi` (aQsi online), ~`atol` (АТОЛ Онлайн), ~`business_ru` (Бизнес.ру), ~`digital_kassa` (digitalkassa), ~`evotor` (Эвотор), ~`first_ofd` (Первый ОФД), ~`kit_invest` (Кит Инвест), ~`komtet` (КОМТЕТ Касса), ~`life_pay` (LIFE PAY), ~`mertrade` (Mertrade), ~`modul_kassa` (МодульКасса), ~`rocket` (RocketR), ~`shtrih_m` (Orange Data).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
*/
class FiscalizationProvider extends AbstractEnum
{
    /** АТОЛ Онлайн */
    public const ATOL = 'atol';
    /** Бизнес.ру */
    public const BUSINESS_RU = 'business_ru';
    /** Orange Data */
    public const SHTRIH_M = 'shtrih_m';
    /** МодульКасса */
    public const MODUL_KASSA = 'modul_kassa';
    /** Эвотор */
    public const EVOTOR = 'evotor';
    /** Кит Инвест */
    public const KIT_INVEST = 'kit_invest';
    /** aQsi online */
    public const A_QSI = 'a_qsi';
    /** Чеки для самозанятых */
    public const FNS = 'fns';
    /** 54-ФЗ: Чеки от ЮKassa */
    public const AVANPOST = 'avanpost';
    /** Mertrade */
    public const MERTRADE = 'mertrade';
    /** Первый ОФД */
    public const FIRST_OFD = 'first_ofd';
    /** LIFE PAY */
    public const LIFE_PAY = 'life_pay';
    /** RocketR */
    public const ROCKET = 'rocket';
    /** digitalkassa */
    public const DIGITAL_KASSA = 'digital_kassa';
    /** КОМТЕТ Касса */
    public const KOMTET = 'komtet';

    /**
     * Возвращает список доступных значений
     * @return string[]
     */
    protected static array $validValues = [
        self::ATOL => true,
        self::AVANPOST => true,
        self::A_QSI => true,
        self::BUSINESS_RU => true,
        self::DIGITAL_KASSA => true,
        self::EVOTOR => true,
        self::FIRST_OFD => true,
        self::FNS => true,
        self::KIT_INVEST => true,
        self::KOMTET => true,
        self::LIFE_PAY => true,
        self::MERTRADE => true,
        self::MODUL_KASSA => true,
        self::ROCKET => true,
        self::SHTRIH_M => true,
    ];
}

