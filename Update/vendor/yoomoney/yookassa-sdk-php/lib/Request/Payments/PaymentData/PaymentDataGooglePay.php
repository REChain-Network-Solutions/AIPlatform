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

namespace YooKassa\Request\Payments\PaymentData;

use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentDataGooglePay.
 *
 * Платежные данные для проведения оплаты при помощи Google Pay.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $paymentMethodToken Криптограмма Payment Token Cryptography для проведения оплаты через Google Pay
 * @property string $payment_method_token Криптограмма Payment Token Cryptography для проведения оплаты через Google Pay
 * @property string $googleTransactionId Уникальный идентификатор транзакции, выданный Google
 * @property string $google_transaction_id Уникальный идентификатор транзакции, выданный Google
 */
class PaymentDataGooglePay extends AbstractPaymentData
{
    /**
     * @var string|null Криптограмма Payment Token Cryptography для проведения оплаты через Google Pay
     */
    #[Assert\NotBlank()]
    #[Assert\Type('string')]
    private ?string $_payment_method_token = null;

    /**
     * @var string|null Уникальный идентификатор транзакции, выданный Google
     */
    #[Assert\NotBlank()]
    #[Assert\Type('string')]
    private ?string $_google_transaction_id = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(PaymentMethodType::GOOGLE_PAY);
    }

    /**
     * Возвращает криптограмму Payment Token Cryptography для проведения оплаты через Google Pay.
     *
     * @return string|null Криптограмма Payment Token Cryptography для проведения оплаты через Google Pay
     */
    public function getPaymentMethodToken(): ?string
    {
        return $this->_payment_method_token;
    }

    /**
     * Устанавливает криптограмму Payment Token Cryptography для проведения оплаты через Google Pay.
     *
     * @param string|null $payment_method_token Криптограмма Payment Token Cryptography для проведения оплаты через Google Pay
     *
     * @return self
     */
    public function setPaymentMethodToken(?string $payment_method_token = null): self
    {
        $this->_payment_method_token = $this->validatePropertyValue('_payment_method_token', $payment_method_token);
        return $this;
    }

    /**
     * Возвращает уникальный идентификатор транзакции, выданный Google.
     *
     * @return string|null Уникальный идентификатор транзакции, выданный Google
     */
    public function getGoogleTransactionId(): ?string
    {
        return $this->_google_transaction_id;
    }

    /**
     * Устанавливает уникальный идентификатор транзакции, выданный Google.
     *
     * @param string|null $google_transaction_id Уникальный идентификатор транзакции, выданный Google
     *
     * @return self
     */
    public function setGoogleTransactionId(?string $google_transaction_id = null): self
    {
        $this->_google_transaction_id = $this->validatePropertyValue('_google_transaction_id', $google_transaction_id);
        return $this;
    }
}
