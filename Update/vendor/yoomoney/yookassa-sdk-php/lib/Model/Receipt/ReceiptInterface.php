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

use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;

/**
 * Interface ReceiptInterface.
 *
 * @category Interface
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property ReceiptCustomerInterface $customer Информация о плательщике
 * @property ReceiptItemInterface[] $items Список товаров в заказе
 * @property int $taxSystemCode Код системы налогообложения. Число 1-6.
 * @property int $tax_system_code Код системы налогообложения. Число 1-6.
 */
interface ReceiptInterface
{
    /**
     * Возвращает Id объекта чека.
     *
     * @return null|string Id объекта чека
     */
    public function getObjectId(): ?string;

    /**
     * Возвращает информацию о плательщике.
     *
     * @return ReceiptCustomerInterface Информация о плательщике
     */
    public function getCustomer(): ReceiptCustomerInterface;

    /**
     * Возвращает список позиций в текущем чеке.
     *
     * @return ReceiptItemInterface[]|ListObjectInterface Список товаров в заказе
     */
    public function getItems(): ListObjectInterface;

    /**
     * Возвращает массив оплат, обеспечивающих выдачу товара.
     *
     * @return SettlementInterface[]|ListObjectInterface Массив оплат, обеспечивающих выдачу товара
     */
    public function getSettlements(): ListObjectInterface;

    /**
     * Возвращает код системы налогообложения.
     *
     * @return int|null Код системы налогообложения. Число 1-6.
     */
    public function getTaxSystemCode(): ?int;

    /**
     * Проверяет есть ли в чеке хотя бы одна позиция.
     *
     * @return bool True если чек не пуст, false если в чеке нет ни одной позиции
     */
    public function notEmpty(): bool;

    /**
     * Подгоняет стоимость товаров в чеке к общей цене заказа.
     *
     * @param AmountInterface $orderAmount Общая стоимость заказа
     * @param bool $withShipping Поменять ли заодно и цену доставки
     */
    public function normalize(AmountInterface $orderAmount, bool $withShipping = false);
}
