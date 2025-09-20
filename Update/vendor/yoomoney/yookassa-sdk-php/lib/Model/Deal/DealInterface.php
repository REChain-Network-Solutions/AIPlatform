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

namespace YooKassa\Model\Deal;

use DateTime;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Metadata;

/**
 * Interface DealInterface.
 *
 * @category Interface
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $id Идентификатор сделки
 * @property string $status Статус сделки
 * @property string $type Тип сделки
 * @property string $fee_moment Момент перечисления вознаграждения
 * @property string $feeMoment Момент перечисления вознаграждения
 * @property string $description Описание сделки
 * @property AmountInterface $balance Баланс сделки
 * @property AmountInterface $payout_balance Сумма вознаграждения продавца
 * @property AmountInterface $payoutBalance Сумма вознаграждения продавца
 * @property DateTime $created_at Время создания сделки
 * @property DateTime $createdAt Время создания сделки
 * @property DateTime $expires_at Время автоматического закрытия сделки
 * @property DateTime $expiresAt Время автоматического закрытия сделки
 * @property Metadata $metadata Дополнительные данные сделки
 * @property bool $test Признак тестовой операции
 */
interface DealInterface
{
    /**
     * Возвращает Id сделки.
     *
     * @return string|null Id сделки
     */
    public function getId(): ?string;

    /**
     * Возвращает тип сделки.
     *
     * @return string|null Тип сделки
     */
    public function getType(): ?string;

    /**
     * Возвращает момент перечисления вам вознаграждения платформы.
     *
     * @return string|null Момент перечисления вознаграждения
     */
    public function getFeeMoment(): ?string;

    /**
     * Возвращает описание сделки (не более 128 символов).
     *
     * @return string|null Описание сделки
     */
    public function getDescription(): ?string;

    /**
     * Возвращает баланс сделки.
     *
     * @return AmountInterface|null Баланс сделки
     */
    public function getBalance(): ?AmountInterface;

    /**
     * Возвращает сумму вознаграждения продавца.
     *
     * @return AmountInterface|null Сумма вознаграждения продавца
     */
    public function getPayoutBalance(): ?AmountInterface;

    /**
     * Возвращает статус сделки.
     *
     * @return string|null Статус сделки
     */
    public function getStatus(): ?string;

    /**
     * Возвращает время создания сделки.
     *
     * @return DateTime|null Время создания сделки
     */
    public function getCreatedAt(): ?DateTime;

    /**
     * Возвращает время автоматического закрытия сделки.
     *
     * @return DateTime|null Время автоматического закрытия сделки
     */
    public function getExpiresAt(): ?DateTime;

    /**
     * Возвращает дополнительные данные сделки.
     *
     * @return Metadata|null Дополнительные данные сделки
     */
    public function getMetadata(): ?Metadata;

    /**
     * Возвращает признак тестовой операции.
     *
     * @return bool|null Признак тестовой операции
     */
    public function getTest(): ?bool;
}
