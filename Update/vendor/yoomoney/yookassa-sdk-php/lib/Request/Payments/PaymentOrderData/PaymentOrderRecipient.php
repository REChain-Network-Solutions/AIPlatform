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

namespace YooKassa\Request\Payments\PaymentOrderData;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentOrderRecipient.
 *
 * Получатель платежа
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $name Название получателя.
 * @property string $inn ИНН получателя.
 * @property string $kpp КПП получателя.
 * @property PaymentOrderRecipientBank $bank Банк получателя.
 */
class PaymentOrderRecipient extends AbstractObject
{
    /**
     * Название получателя.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_name = null;

    /**
     * ИНН получателя.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex("/^[0-9]{10}$/")]
    private ?string $_inn = null;

    /**
     * КПП получателя.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex("/^[0-9]{9}$/")]
    private ?string $_kpp = null;

    /**
     * Банк получателя.
     *
     * @var PaymentOrderRecipientBank|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(PaymentOrderRecipientBank::class)]
    private ?PaymentOrderRecipientBank $_bank = null;

    /**
     * Возвращает name.
     *
     * @return string|null
     */
    public function getName(): ?string
    {
        return $this->_name;
    }

    /**
     * Устанавливает name.
     *
     * @param string|null $name Название получателя.
     *
     * @return self
     */
    public function setName(?string $name = null): self
    {
        $this->_name = $this->validatePropertyValue('_name', $name);
        return $this;
    }

    /**
     * Возвращает inn.
     *
     * @return string|null
     */
    public function getInn(): ?string
    {
        return $this->_inn;
    }

    /**
     * Устанавливает inn.
     *
     * @param string|null $inn ИНН получателя.
     *
     * @return self
     */
    public function setInn(?string $inn = null): self
    {
        $this->_inn = $this->validatePropertyValue('_inn', $inn);
        return $this;
    }

    /**
     * Возвращает kpp.
     *
     * @return string|null
     */
    public function getKpp(): ?string
    {
        return $this->_kpp;
    }

    /**
     * Устанавливает kpp.
     *
     * @param string|null $kpp КПП получателя.
     *
     * @return self
     */
    public function setKpp(?string $kpp = null): self
    {
        $this->_kpp = $this->validatePropertyValue('_kpp', $kpp);
        return $this;
    }

    /**
     * Возвращает bank.
     *
     * @return PaymentOrderRecipientBank|null
     */
    public function getBank(): ?PaymentOrderRecipientBank
    {
        return $this->_bank;
    }

    /**
     * Устанавливает bank.
     *
     * @param PaymentOrderRecipientBank|array|null $bank Банк получателя.
     *
     * @return self
     */
    public function setBank(mixed $bank = null): self
    {
        $this->_bank = $this->validatePropertyValue('_bank', $bank);
        return $this;
    }
}
