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

namespace YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель ElectronicCertificatePaymentData.
 *
 * Данные от ФЭС НСПК для оплаты по электронному сертификату.
 * Необходимо передавать только при [оплате со сбором данных на вашей стороне](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/merchant-payment-form).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property AmountInterface $amount Сумма, которая спишется с электронного сертификата.
 * @property string $basket_id Идентификатор корзины покупки, сформированной в НСПК.
 * @property string $basketId Идентификатор корзины покупки, сформированной в НСПК.
 */
class ElectronicCertificatePaymentData extends AbstractObject
{
    /**
     * Сумма, которую необходимо использовать по электронному сертификату, — значение `totalCertAmount`, которое вы получили в ФЭС НСПК в [запросе на предварительное одобрение использования сертификата (Pre-Auth)](https://www.nspk.ru/developer/api-fes#operation/preAuthPurchase).
     * Сумма должна быть не больше общей суммы платежа (`amount`).
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_amount = null;

    /**
     * Идентификатор корзины покупки, сформированной в НСПК, — значение `purchaseBasketId`, которое вы получили в ФЭС НСПК в [запросе на предварительное одобрение использования сертификата (Pre-Auth)](https://www.nspk.ru/developer/api-fes#operation/preAuthPurchase).
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_basket_id = null;

    /**
     * Возвращает amount.
     *
     * @return AmountInterface|null Сумма, которая спишется с электронного сертификата
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает amount.
     *
     * @param AmountInterface|array|null $amount Сумма, которая спишется с электронного сертификата
     *
     * @return self
     */
    public function setAmount(mixed $amount = null): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }

    /**
     * Возвращает basket_id.
     *
     * @return string|null Идентификатор корзины покупки
     */
    public function getBasketId(): ?string
    {
        return $this->_basket_id;
    }

    /**
     * Устанавливает basket_id.
     *
     * @param string|null $basket_id Идентификатор корзины покупки
     *
     * @return self
     */
    public function setBasketId(?string $basket_id = null): self
    {
        $this->_basket_id = $this->validatePropertyValue('_basket_id', $basket_id);
        return $this;
    }

}

