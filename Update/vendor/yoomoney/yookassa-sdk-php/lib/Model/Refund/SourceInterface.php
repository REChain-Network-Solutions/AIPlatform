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

namespace YooKassa\Model\Refund;

use YooKassa\Model\AmountInterface;

/**
 * Interface SourceInterface.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property AmountInterface $amount Сумма возврата
 * @property AmountInterface $platformFeeAmount Комиссия, которую вы удержали при оплате, и хотите вернуть
 * @property AmountInterface $platform_fee_amount Комиссия, которую вы удержали при оплате, и хотите вернуть
 * @property string $accountId Идентификатор магазина, для которого вы хотите провести возврат
 * @property string $account_id Идентификатор магазина, для которого вы хотите провести возврат
 */
interface SourceInterface
{
    /**
     * Устанавливает id магазина-получателя средств.
     *
     * @param string|null $account_id Id магазина с которого будут списаны средства
     */
    public function setAccountId(?string $account_id): SourceInterface;

    /**
     * Возвращает id магазина с которого будут списаны средства.
     *
     * @return string|null Id магазина с которого будут списаны средства
     */
    public function getAccountId(): ?string;

    /**
     * Возвращает сумму оплаты.
     *
     * @return AmountInterface|null Сумма оплаты
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
     * @param AmountInterface|array|null $amount Сумма оплаты
     */
    public function setAmount(mixed $amount = null): SourceInterface;

    /**
     * Возвращает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу.
     *
     * @return AmountInterface|null Сумма комиссии
     */
    public function getPlatformFeeAmount(): ?AmountInterface;

    /**
     * Проверяет, была ли установлена комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу.
     *
     * @return bool True если комиссия была установлена, false если нет
     */
    public function hasPlatformFeeAmount(): bool;

    /**
     * Устанавливает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу.
     *
     * @param AmountInterface|array|null $platform_fee_amount Сумма комиссии
     *
     * @return SourceInterface
     */
    public function setPlatformFeeAmount(mixed $platform_fee_amount): SourceInterface;
}
