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

namespace YooKassa\Model\Payout;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\Payment\PaymentMethod\BankCardType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PayoutDestinationBankCardCard.
 *
 * Данные банковской карты.
 * Необходим при оплате PCI-DSS данными.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $last4 Последние 4 цифры номера карты
 * @property string $first6 Первые 6 цифр номера карты
 * @property string $cardType Тип банковской карты
 * @property string $card_type Тип банковской карты
 * @property string $issuerCountry Код страны, в которой выпущена карта
 * @property string $issuer_country Код страны, в которой выпущена карта
 * @property string $issuerName Тип банковской карты
 * @property string $issuer_name Тип банковской карты
 */
class PayoutDestinationBankCardCard extends AbstractObject
{
    /** @var int Длина кода страны по ISO 3166 https://www.iso.org/obp/ui/#iso:pub:PUB500001:en */
    public const ISO_3166_CODE_LENGTH = 2;

    /**
     * Первые 6 цифр номера карты (BIN).
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex("/^[0-9]{6}$/")]
    private ?string $_first6 = null;

    /**
     * Последние 4 цифры номера карты.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex("/^[0-9]{4}$/")]
    private ?string $_last4 = null;

    /**
     * Тип банковской карты.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [BankCardType::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_card_type = null;

    /**
     * Код страны, в которой выпущена карта. Пример: ~`RU`.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Regex("/^[A-Z]{2}$/")]
    private ?string $_issuer_country = null;

    /**
     * Наименование банка, выпустившего карту.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_issuer_name = null;

    /**
     * Возвращает первые 6 цифр номера карты.
     *
     * @return string|null Первые 6 цифр номера карты
     */
    public function getFirst6(): ?string
    {
        return $this->_first6;
    }

    /**
     * Устанавливает первые 6 цифр номера карты.
     *
     * @param string|null $first6 Первые 6 цифр номера карты
     *
     * @return self
     */
    public function setFirst6(?string $first6 = null): self
    {
        $this->_first6 = $this->validatePropertyValue('_first6', $first6);
        return $this;
    }

    /**
     * Возвращает последние 4 цифры номера карты.
     *
     * @return string|null Последние 4 цифры номера карты
     */
    public function getLast4(): ?string
    {
        return $this->_last4;
    }

    /**
     * Устанавливает последние 4 цифры номера карты.
     *
     * @param string|null $last4 Последние 4 цифры номера карты
     *
     * @return self
     */
    public function setLast4(?string $last4 = null): self
    {
        $this->_last4 = $this->validatePropertyValue('_last4', $last4);
        return $this;
    }

    /**
     * Возвращает тип банковской карты.
     *
     * @return string|null Тип банковской карты
     */
    public function getCardType(): ?string
    {
        return $this->_card_type;
    }

    /**
     * Устанавливает тип банковской карты.
     *
     * @param string|null $card_type Тип банковской карты
     *
     * @return self
     */
    public function setCardType(?string $card_type = null): self
    {
        $this->_card_type = $this->validatePropertyValue('_card_type', $card_type);
        return $this;
    }

    /**
     * Возвращает код страны, в которой выпущена карта. Передается в формате ISO-3166 alpha-2.
     *
     * @return string|null Код страны, в которой выпущена карта
     */
    public function getIssuerCountry(): ?string
    {
        return $this->_issuer_country;
    }

    /**
     * Устанавливает код страны, в которой выпущена карта. Передается в формате ISO-3166 alpha-2.
     *
     * @param string|null $issuer_country Код страны, в которой выпущена карта
     *
     * @return self
     */
    public function setIssuerCountry(?string $issuer_country = null): self
    {
        $this->_issuer_country = $this->validatePropertyValue('_issuer_country', $issuer_country);
        return $this;
    }

    /**
     * Возвращает наименование банка, выпустившего карту.
     *
     * @return string|null Наименование банка, выпустившего карту
     */
    public function getIssuerName(): ?string
    {
        return $this->_issuer_name;
    }

    /**
     * Устанавливает наименование банка, выпустившего карту.
     *
     * @param string|null $issuer_name Наименование банка, выпустившего карту
     *
     * @return self
     */
    public function setIssuerName(?string $issuer_name = null): self
    {
        $this->_issuer_name = $this->validatePropertyValue('_issuer_name', $issuer_name);
        return $this;
    }
}
