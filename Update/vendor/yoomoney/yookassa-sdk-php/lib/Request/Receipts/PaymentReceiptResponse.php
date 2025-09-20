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

use YooKassa\Validator\Constraints as Assert;

/**
 * Класс описывающий чек, привязанный к платежу.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $payment_id Идентификатор платежа в ЮKassa
 * @property string $paymentId Идентификатор платежа в ЮKassa
 */
class PaymentReceiptResponse extends AbstractReceiptResponse
{
    /** Длина идентификатора платежа */
    public const LENGTH_PAYMENT_ID = 36;

    /**
     * @var string|null Идентификатор платежа
     */
    #[Assert\Type('string')]
    #[Assert\Length(exactly: self::LENGTH_PAYMENT_ID)]
    private ?string $_payment_id = null;

    /**
     * Установка свойств, присущих конкретному объекту.
     */
    public function setSpecificProperties(array $receiptData): void
    {
        $this->setPaymentId($this->getObjectId());
    }

    /**
     * Возвращает идентификатор платежа.
     *
     * @return string|null Идентификатор платежа
     */
    public function getPaymentId(): ?string
    {
        return $this->_payment_id;
    }

    /**
     * Устанавливает идентификатор платежа в ЮKassa.
     *
     * @param string|null $payment_id Идентификатор платежа в ЮKassa
     *
     *@return self
     */
    public function setPaymentId(?string $payment_id = null): self
    {
        $this->_payment_id = $this->validatePropertyValue('_payment_id', $payment_id);
        return $this;
    }
}
