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

use YooKassa\Model\AmountInterface;
use YooKassa\Model\Metadata;

/**
 * Interface TransferDataInterface.
 *
 * Данные о распределении денег — сколько и в какой магазин нужно перевести.
 * Присутствует, если вы используете [Сплитование платежей](/developers/solutions-for-platforms/split-payments/basics).
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $accountId Идентификатор магазина, в пользу которого вы принимаете оплату
 * @property string $account_id Идентификатор магазина, в пользу которого вы принимаете оплату
 * @property AmountInterface $amount Сумма, которую необходимо перечислить магазину
 * @property AmountInterface $platformFeeAmount Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу
 * @property AmountInterface $platform_fee_amount Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу
 * @property string $description Описание транзакции, которое продавец увидит в личном кабинете ЮKassa. (например: «Заказ маркетплейса №72»)
 * @property Metadata $metadata Любые дополнительные данные, которые нужны вам для работы с платежами (например, ваш внутренний идентификатор заказа)
 */
interface TransferDataInterface
{
    /**
     * Возвращает account_id.
     *
     * @return string|null
     */
    public function getAccountId(): ?string;

    /**
     * Устанавливает account_id.
     *
     * @param string|null $value
     *
     * @return TransferData
     */
    public function setAccountId(?string $value = null): TransferData;

    /**
     * @return bool
     */
    public function hasAccountId(): bool;

    /**
     * Возвращает amount.
     *
     * @return AmountInterface|null
     */
    public function getAmount(): ?AmountInterface;

    /**
     * Устанавливает amount.
     *
     * @param AmountInterface|array|null $value
     *
     * @return TransferData
     */
    public function setAmount(mixed $value = null): TransferData;

    /**
     * @return bool
     */
    public function hasAmount(): bool;

    /**
     * Возвращает platform_fee_amount.
     *
     * @return AmountInterface|null
     */
    public function getPlatformFeeAmount(): ?AmountInterface;

    /**
     * Устанавливает platform_fee_amount.
     *
     * @param AmountInterface|array|null $value
     *
     * @return TransferData
     */
    public function setPlatformFeeAmount(mixed $value = null): TransferData;

    /**
     * @return bool
     */
    public function hasPlatformFeeAmount(): bool;

    /**
     * Возвращает description.
     *
     * @return string|null
     */
    public function getDescription(): ?string;

    /**
     * Устанавливает description.
     *
     * @param string|null $value Описание транзакции (не более 128 символов), которое продавец увидит в личном кабинете ЮKassa. Например: «Заказ маркетплейса №72».
     *
     * @return TransferData
     */
    public function setDescription(?string $value = null): TransferData;

    /**
     * @return bool
     */
    public function hasDescription(): bool;

    /**
     * Возвращает metadata.
     *
     * @return Metadata|null
     */
    public function getMetadata(): ?Metadata;

    /**
     * Устанавливает metadata.
     *
     * @param string|array|null $value Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @return TransferData
     */
    public function setMetadata(mixed $value = null): TransferData;

    /**
     * @return bool
     */
    public function hasMetadata(): bool;
}
