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

namespace YooKassa\Request\PaymentMethods;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentMethodCard.
 *
 * Данные банковской карты (необходимы, если вы собираете данные карты пользователей на своей стороне).
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $number Номер банковской карты.
 * @property string $expiryYear Срок действия, год, YYYY.
 * @property string $expiry_year Срок действия, год, YYYY.
 * @property string $expiryMonth Срок действия, месяц, MM.
 * @property string $expiry_month Срок действия, месяц, MM.
 * @property string $csc Код CVC2 или CVV2, 3 или 4 символа, печатается на обратной стороне карты.
 * @property string $cardholder Имя владельца карты.
 */
class PaymentMethodCard extends AbstractObject
{
    /**
     * @var string|null Номер банковской карты
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex('/^[0-9]{14,19}$/')]
    private ?string $_number = null;

    /**
     * @var string|null Срок действия, год, YY
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex('/^[0-9]{4}$/')]
    private ?string $_expiry_year = null;

    /**
     * @var string|null Срок действия, месяц, MM
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex("/^(0?[1-9]|1[0-2])$/")]
    #[Assert\GreaterThanOrEqual(value: 1)]
    #[Assert\LessThanOrEqual(value: 12)]
    private ?string $_expiry_month = null;

    /**
     * @var string|null CVV2/CVC2 код
     */
    #[Assert\Type('string')]
    #[Assert\Regex('/^[0-9]{3,4}$/')]
    private ?string $_csc = null;

    /**
     * @var string|null Имя держателя карты
     */
    #[Assert\Type('string')]
    #[Assert\Regex('/^[a-zA-Z \'-]{1,26}$/')]
    private ?string $_cardholder = null;

    /**
     * Возвращает номер банковской карты.
     *
     * @return string|null Номер банковской карты
     */
    public function getNumber(): ?string
    {
        return $this->_number;
    }

    /**
     * Устанавливает номер банковской карты.
     *
     * @param string|null $number Номер банковской карты
     *
     * @return self
     */
    public function setNumber(?string $number = null): self
    {
        $this->_number = $this->validatePropertyValue('_number', $number);
        return $this;
    }

    /**
     * Возвращает год срока действия карты.
     *
     * @return string|null Срок действия, год, YYYY
     */
    public function getExpiryYear(): ?string
    {
        return $this->_expiry_year;
    }

    /**
     * Устанавливает год срока действия карты.
     *
     * @param string|null $expiry_year Срок действия, год, YYYY
     *
     * @return self
     */
    public function setExpiryYear(?string $expiry_year = null): self
    {
        $this->_expiry_year = $this->validatePropertyValue('_expiry_year', $expiry_year);
        return $this;
    }

    /**
     * Возвращает месяц срока действия карты.
     *
     * @return string|null Срок действия, месяц, MM
     */
    public function getExpiryMonth(): ?string
    {
        return $this->_expiry_month;
    }

    /**
     * Устанавливает месяц срока действия карты.
     *
     * @param string|null $expiry_month Срок действия, месяц, MM
     *
     * @return self
     */
    public function setExpiryMonth(?string $expiry_month = null): self
    {
        $this->_expiry_month = $this->validatePropertyValue('_expiry_month', $expiry_month);
        return $this;
    }

    /**
     * Возвращает CVV2/CVC2 код.
     *
     * @return string|null CVV2/CVC2 код
     */
    public function getCsc(): ?string
    {
        return $this->_csc;
    }

    /**
     * Устанавливает CVV2/CVC2 код.
     *
     * @param string|null $csc CVV2/CVC2 код
     *
     * @return self
     */
    public function setCsc(?string $csc = null): self
    {
        $this->_csc = $this->validatePropertyValue('_csc', $csc);
        return $this;
    }

    /**
     * Возвращает имя держателя карты.
     *
     * @return string|null Имя держателя карты
     */
    public function getCardholder(): ?string
    {
        return $this->_cardholder;
    }

    /**
     * Устанавливает имя держателя карты.
     *
     * @param string|null $cardholder Имя держателя карты
     *
     * @return self
     */
    public function setCardholder(?string $cardholder = null): self
    {
        $this->_cardholder = $this->validatePropertyValue('_cardholder', $cardholder);
        return $this;
    }
}
