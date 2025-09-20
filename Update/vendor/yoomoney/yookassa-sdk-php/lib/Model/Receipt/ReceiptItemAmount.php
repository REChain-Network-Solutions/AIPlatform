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

namespace YooKassa\Model\Receipt;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;
use YooKassa\Common\Exceptions\EmptyPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\CurrencyCode;

/**
 * Class ReceiptItemAmount.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property int $value Сумма
 * @property string $currency Код валюты
 */
class ReceiptItemAmount extends AbstractObject implements AmountInterface
{
    /**
     * @var int Сумма
     */
    #[Assert\NotBlank]
    #[Assert\Type('numeric')]
    private ?int $_value = 0;

    /**
     * @var string Код валюты
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [CurrencyCode::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_currency = CurrencyCode::RUB;

    /**
     * MonetaryAmount constructor.
     *
     * @param null|array|numeric $value Сумма
     * @param null|string $currency Код валюты
     */
    public function __construct($value = null, ?string $currency = null)
    {
        if (is_array($value)) {
            parent::__construct($value);
        } else {
            parent::__construct();
            if (null !== $value && $value > 0.0) {
                $this->setValue($value);
            }
            if (null !== $currency) {
                $this->setCurrency($currency);
            }
        }
    }

    /**
     * Возвращает значение суммы.
     *
     * @return string Сумма
     */
    public function getValue(): string
    {
        if ($this->_value < 10) {
            return '0.0' . $this->_value;
        }
        if ($this->_value < 100) {
            return '0.' . $this->_value;
        }

        return substr($this->_value, 0, -2) . '.' . substr($this->_value, -2);
    }

    /**
     * {@inheritdoc}
     */
    public function setValue(mixed $value): void
    {
        $value = $this->validatePropertyValue('_value', $value);
        if ($value < 0.0) {
            throw new InvalidPropertyValueException('Invalid amount value: "' . $value . '"', 0, 'amount.value', $value);
        }
        $castedValue = (int) round($value * 100.0);

        $this->_value = $castedValue;
    }

    /**
     * Возвращает сумму в копейках в виде целого числа.
     *
     * @return int Сумма в копейках/центах
     */
    public function getIntegerValue(): int
    {
        return $this->_value;
    }

    /**
     * Возвращает валюту.
     *
     * @return string Код валюты
     */
    public function getCurrency(): string
    {
        return $this->_currency;
    }

    /**
     * Устанавливает код валюты.
     *
     * @param string|null $currency Код валюты
     *
     * @return self
     */
    public function setCurrency(?string $currency = null): self
    {
        $this->_currency = $this->validatePropertyValue('_currency', $currency);
        return $this;
    }

    /**
     * Умножает текущую сумму на указанный коэффициент
     *
     * @param float|null $coefficient Множитель
     *
     * @throws EmptyPropertyValueException Выбрасывается если передано пустое значение
     * @throws InvalidPropertyValueTypeException Выбрасывается если было передано не число
     * @throws InvalidPropertyValueException Выбрасывается если переданное значение меньше или равно нулю, либо если
     *                                       после умножения получили значение равное нулю
     */
    public function multiply(?float $coefficient): void
    {
        if (null === $coefficient || '' === $coefficient) {
            throw new EmptyPropertyValueException('Empty coefficient in multiply method', 0, 'amount.value');
        }
        if (!is_numeric($coefficient)) {
            throw new InvalidPropertyValueTypeException(
                'Invalid coefficient type in multiply method',
                0,
                'amount.value',
                $coefficient
            );
        }
        if ($coefficient <= 0.0) {
            throw new InvalidPropertyValueException(
                'Invalid coefficient in multiply method: "' . $coefficient . '"',
                0,
                'amount.value',
                $coefficient
            );
        }
        $castedValue = (int) round($coefficient * $this->_value);
        if (0 === $castedValue) {
            throw new InvalidPropertyValueException(
                'Invalid coefficient value in multiply method: "' . $coefficient . '"',
                0,
                'amount.value',
                $coefficient
            );
        }
        $this->_value = $castedValue;
    }

    /**
     * Увеличивает сумму на указанное значение.
     *
     * @param float|null $value Значение, которое будет прибавлено к текущему
     *
     * @throws EmptyPropertyValueException Выбрасывается если передано пустое значение
     * @throws InvalidPropertyValueTypeException Выбрасывается если было передано не число
     * @throws InvalidPropertyValueException Выбрасывается если после сложения получилась сумма меньше или равная нулю
     */
    public function increase(?float $value): void
    {
        if (null === $value) {
            throw new EmptyPropertyValueException('Empty amount value in increase method', 0, 'amount.value');
        }
        if (!is_numeric($value)) {
            throw new InvalidPropertyValueTypeException(
                'Invalid amount value type in increase method',
                0,
                'amount.value',
                $value
            );
        }
        $castedValue = (int) round($this->_value + $value * 100.0);
        if ($castedValue <= 0) {
            throw new InvalidPropertyValueException(
                'Invalid amount value in increase method: "' . $value . '"',
                0,
                'amount.value',
                $value
            );
        }
        $this->_value = $castedValue;
    }

    public function jsonSerialize(): array
    {
        return [
            'value' => number_format($this->_value / 100.0, 2, '.', ''),
            'currency' => $this->_currency,
        ];
    }
}
