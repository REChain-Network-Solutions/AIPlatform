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

namespace YooKassa\Model\Payment;

use YooKassa\Model\AmountInterface;
use YooKassa\Model\Metadata;

/**
 * Interface TransferInterface.
 *
 * Данные о распределении денег — сколько и в какой магазин нужно перевести.
 * Присутствует, если вы используете [Сплитование платежей](/developers/solutions-for-platforms/split-payments/basics).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $accountId Идентификатор магазина, в пользу которого вы принимаете оплату
 * @property string $account_id Идентификатор магазина, в пользу которого вы принимаете оплату
 * @property AmountInterface $amount Сумма, которую необходимо перечислить магазину
 * @property AmountInterface $platformFeeAmount Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу
 * @property AmountInterface $platform_fee_amount Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу
 * @property string $status Статус распределения денег между магазинами. Возможные значения: `pending`, `waiting_for_capture`, `succeeded`, `canceled`
 * @property string $description Описание транзакции, которое продавец увидит в личном кабинете ЮKassa. (например: «Заказ маркетплейса №72»)
 * @property Metadata $metadata Любые дополнительные данные, которые нужны вам для работы с платежами (например, ваш внутренний идентификатор заказа)
 * @property bool $releaseFunds Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать
 * @property bool $release_funds Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать
 * @property string $connectedAccountId Идентификатор продавца, подключенного к вашей платформе
 * @property string $connected_account_id Идентификатор продавца, подключенного к вашей платформе
 */
interface TransferInterface
{
    /**
     * Устанавливает идентификатор магазина-получателя средств.
     *
     * @param string $value Идентификатор магазина-получателя средств
     */
    public function setAccountId(string $value): self;

    /**
     * Возвращает идентификатор магазина-получателя средств.
     *
     * @return null|string Идентификатор магазина-получателя средств
     */
    public function getAccountId(): ?string;

    /**
     * Возвращает сумму оплаты.
     *
     * @return AmountInterface|null Сумма оплаты
     */
    public function getAmount(): ?AmountInterface;

    /**
     * Устанавливает сумму оплаты.
     *
     * @param AmountInterface|array|null $value Сумма оплаты
     */
    public function setAmount(mixed $value): self;

    /**
     * Возвращает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу.
     *
     * @return AmountInterface|null Сумма комиссии
     */
    public function getPlatformFeeAmount(): ?AmountInterface;

    /**
     * Устанавливает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу.
     *
     * @param AmountInterface|array|null $value Сумма комиссии
     */
    public function setPlatformFeeAmount(mixed $value): self;

    /**
     * Возвращает статус операции распределения средств конечному получателю.
     *
     * @return null|string Статус операции распределения средств конечному получателю
     */
    public function getStatus(): ?string;

    /**
     * Устанавливает статус операции распределения средств конечному получателю.
     */
    public function setStatus(?string $value);

    /**
     * Устанавливает описание транзакции.
     *
     * @param null|string $value Описание транзакции
     */
    public function setDescription(?string $value): self;

    /**
     * Возвращает описание транзакции.
     *
     * @return null|string Описание транзакции
     */
    public function getDescription(): ?string;

    /**
     * Устанавливает метаданные.
     *
     * @param null|array|Metadata $value Метаданные
     */
    public function setMetadata(mixed $value): self;

    /**
     * Возвращает метаданные.
     *
     * @return null|Metadata Метаданные
     */
    public function getMetadata(): ?Metadata;
}
