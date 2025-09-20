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

namespace YooKassa\Request\Receipts;

use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Receipt\SettlementInterface;

/**
 * Interface ReceiptInterface.
 *
 * @category Interface
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $id Идентификатор чека в ЮKassa.
 * @property string $type Тип чека в онлайн-кассе: приход "payment" или возврат "refund".
 * @property string $receiptRegistration Статус доставки данных для чека в онлайн-кассу ("pending", "succeeded" или "canceled").
 * @property string $receipt_registration Статус доставки данных для чека в онлайн-кассу ("pending", "succeeded" или "canceled").
 * @property string $status Статус доставки данных для чека в онлайн-кассу ("pending", "succeeded" или "canceled").
 * @property int $taxSystemCode Код системы налогообложения. Число 1-6.
 * @property int $tax_system_code Код системы налогообложения. Число 1-6.
 * @property ReceiptResponseItemInterface[] $items Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров.
 * @property SettlementInterface[] $settlements Перечень совершенных расчетов.
 */
interface ReceiptResponseInterface
{
    /**
     * Возвращает идентификатор чека в ЮKassa.
     *
     * @return string|null Идентификатор чека в ЮKassa
     */
    public function getId(): ?string;

    /**
     * Возвращает тип чека в онлайн-кассе.
     *
     * @return string|null Тип чека в онлайн-кассе: приход "payment" или возврат "refund"
     */
    public function getType(): ?string;

    /**
     * Возвращает статус доставки данных для чека в онлайн-кассу.
     *
     * @return string|null Статус доставки данных для чека в онлайн-кассу ("pending", "succeeded" или "canceled")
     */
    public function getStatus(): ?string;

    /**
     * Возвращает код системы налогообложения.
     *
     * @return int|null Код системы налогообложения. Число 1-6.
     */
    public function getTaxSystemCode(): ?int;

    /**
     * Возвращает Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров.
     *
     *  @return ReceiptResponseItemInterface[]|ListObjectInterface
     */
    public function getItems(): ListObjectInterface;

    /**
     * Возвращает список расчетов.
     *
     *  @return SettlementInterface[]|ListObjectInterface
     */
    public function getSettlements(): ListObjectInterface;

    /**
     * Возвращает идентификатор магазин
     *
     * @return string|null
     */
    public function getOnBehalfOf(): ?string;

    /**
     * Проверяет есть ли в чеке хотя бы одна позиция.
     *
     * @return bool True если чек не пуст, false если в чеке нет ни одной позиции
     */
    public function notEmpty(): bool;
}
