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

namespace YooKassa\Request\Payments;

use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Deal\CaptureDealData;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Receipt\ReceiptInterface;

/**
 * Interface CreateCaptureRequestInterface.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property MonetaryAmount $amount Подтверждаемая сумма оплаты
 * @property ReceiptInterface $receipt Данные фискального чека 54-ФЗ
 * @property CaptureDealData $deal Данные о сделке, в составе которой проходит платеж
 */
interface CreateCaptureRequestInterface
{
    /**
     * Возвращает подтверждаемую сумму оплаты.
     *
     * @return AmountInterface|null Подтверждаемая сумма оплаты
     */
    public function getAmount(): ?AmountInterface;

    /**
     * Проверяет, была ли установлена сумма оплаты.
     *
     * @return bool True если сумма оплаты была установлена, false если нет
     */
    public function hasAmount(): bool;

    /**
     * Устанавливает сумму оплаты.
     *
     * @param AmountInterface|array|string $amount Сумма оплаты
     */
    public function setAmount(mixed $amount);

    /**
     * Возвращает чек, если он есть.
     *
     * @return null|ReceiptInterface Данные фискального чека 54-ФЗ или null, если чека нет
     */
    public function getReceipt(): ?ReceiptInterface;

    /**
     * Проверяет наличие чека в создаваемом платеже.
     *
     * @return bool True если чек есть, false если нет
     */
    public function hasReceipt(): bool;

    /**
     * Устанавливает чек.
     *
     * @param null|ReceiptInterface $value Инстанс чека или null для удаления информации о чеке
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если передан не инстанс класса чека и не null
     */
    public function setReceipt(?ReceiptInterface $value): AbstractRequestInterface;

    /**
     * Проверяет наличие данных о распределении денег.
     */
    public function hasTransfers(): bool;

    /**
     * Возвращает данные о распределении денег.
     *
     * @return TransferDataInterface[]|ListObjectInterface|null
     */
    public function getTransfers(): ?ListObjectInterface;

    /**
     * Устанавливает transfers (массив распределения денег между магазинами).
     *
     * @param null|array|TransferDataInterface[] $transfers
     *
     * @return self
     */
    public function setTransfers(mixed $transfers = null): AbstractRequestInterface;

    /**
     * Проверяет наличие данных о сделке.
     */
    public function hasDeal(): bool;

    /**
     * Возвращает данные о сделке.
     */
    public function getDeal(): ?CaptureDealData;

    /**
     * Устанавливает данные о сделке.
     *
     * @param null|array|CaptureDealData $deal
     */
    public function setDeal(mixed $deal): AbstractRequestInterface;
}
