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
 * Класс, представляющий модель PaymentDataApplePay.
 *
 * Платежные данные для проведения оплаты при помощи Apple Pay.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $paymentData содержимое поля paymentData объекта PKPaymentToken, закодированное в Base64
 * @property string $payment_data содержимое поля paymentData объекта PKPaymentToken, закодированное в Base64
 */
class PaymentDataApplePay extends AbstractPaymentData
{
    /**
     * @var string|null Содержимое поля paymentData объекта PKPaymentToken, закодированное в Base64
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_payment_data = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(PaymentMethodType::APPLE_PAY);
    }

    /**
     * Возвращает содержимое поля paymentData объекта PKPaymentToken, закодированное в Base64.
     *
     * @return string|null содержимое поля paymentData объекта PKPaymentToken, закодированное в Base64
     */
    public function getPaymentData(): ?string
    {
        return $this->_payment_data;
    }

    /**
     * Устанавливает содержимое поля paymentData объекта PKPaymentToken, закодированное в Base64.
     *
     * @param string|null $payment_data содержимое поля paymentData объекта PKPaymentToken, закодированное в Base64
     *
     * @return self
     */
    public function setPaymentData(?string $payment_data): self
    {
        $this->_payment_data = $this->validatePropertyValue('_payment_data', $payment_data);
        return $this;
    }
}
