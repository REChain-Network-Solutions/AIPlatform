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

namespace YooKassa\Request\Refunds;

use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Deal\RefundDealData;
use YooKassa\Model\Receipt\ReceiptInterface;
use YooKassa\Model\Refund\SourceInterface;

/**
 * Interface CreateRefundRequestInterface
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $paymentId Айди платежа для которого создаётся возврат
 * @property AmountInterface $amount Сумма возврата
 * @property string $description Комментарий к операции возврата, основание для возврата средств покупателю.
 * @property null|ReceiptInterface $receipt Инстанс чека или null
 * @property null|SourceInterface[] $sources Информация о распределении денег — сколько и в какой магазин нужно перевести
 * @property null|RefundDealData $deal Информация о сделке
 */
interface CreateRefundRequestInterface
{
    /**
     * Возвращает айди платежа для которого создаётся возврат средств.
     *
     * @return string|null Айди платежа для которого создаётся возврат
     */
    public function getPaymentId(): ?string;

    /**
     * Возвращает сумму возвращаемых средств.
     *
     * @return AmountInterface|null Сумма возврата
     */
    public function getAmount(): ?AmountInterface;

    /**
     * Проверяет, был ли установлена идентификатор платежа.
     *
     * @return bool True если идентификатор платежа был установлен, false если нет
     */
    public function hasPaymentId(): bool;

    /**
     * Устанавливает комментарий к возврату.
     *
     * @param string|null $description Комментарий к операции возврата, основание для возврата средств покупателю
     */
    public function setDescription(?string $description);

    /**
     * Возвращает комментарий к возврату или null, если комментарий не задан
     *
     * @return string|null Комментарий к операции возврата, основание для возврата средств покупателю.
     */
    public function getDescription(): ?string;

    /**
     * Проверяет задан ли комментарий к создаваемому возврату.
     *
     * @return bool True если комментарий установлен, false если нет
     */
    public function hasDescription(): bool;

    /**
     * Устанавливает чек.
     *
     * @param null|ReceiptInterface $receipt Инстанс чека или null для удаления информации о чеке
     */
    public function setReceipt(?ReceiptInterface $receipt);

    /**
     * Возвращает инстанс чека или null, если чек не задан.
     *
     * @return null|ReceiptInterface Инстанс чека или null
     */
    public function getReceipt(): ?ReceiptInterface;

    /**
     * Проверяет задан ли чек.
     *
     * @return bool True если чек есть, false если нет
     */
    public function hasReceipt(): bool;

    /**
     * Устанавливает информацию о распределении денег — сколько и в какой магазин нужно перевести.
     *
     * @param SourceInterface[]|array|null $sources Информация о распределении денег
     */
    public function setSources(?array $sources);

    /**
     * Возвращает информацию о распределении денег — сколько и в какой магазин нужно перевести.
     *
     * @return SourceInterface[]|ListObjectInterface Информация о распределении денег
     */
    public function getSources(): ListObjectInterface;

    /**
     * Проверяет наличие информации о распределении денег.
     */
    public function hasSources(): bool;

    /**
     * Устанавливает информацию о сделке.
     *
     * @param RefundDealData|null $deal Информация о сделке
     */
    public function setDeal(?RefundDealData $deal);

    /**
     * Возвращает информацию о сделке.
     *
     * @return RefundDealData|null Информация о сделке
     */
    public function getDeal(): ?RefundDealData;

    /**
     * Проверяет наличие информации о сделке.
     */
    public function hasDeal(): bool;
}
