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

namespace YooKassa\Model;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;
use YooKassa\Validator\Exceptions\EmptyPropertyValueException;
use YooKassa\Validator\Exceptions\InvalidPropertyValueException;
use YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException;

/**
 * MonetaryAmount - Сумма определенная в валюте.
 *
 * @property int $value Сумма
 * @property string $currency Код валюты
 */
class MonetaryAmount extends AbstractObject implements AmountInterface
{
    /**
     * Сумма в выбранной валюте.
     *
     * @var int|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    private ?int $_value = 0;

    /**
     * Трехбуквенный код валюты в формате ISO-4217. Пример: RUB.
     *
     * @var string|null
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
    public function __construct(mixed $value = null, ?string $currency = null)
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
     * Устанавливает сумму.
     *
     * @param numeric|string|null $value Сумма
     *
     * @throws EmptyPropertyValueException Генерируется если было передано пустое значение
     * @throws InvalidPropertyValueTypeException Генерируется если было передано значение невалидного типа
     * @throws InvalidPropertyValueException Генерируется если было передано не валидное значение
     */
    public function setValue(mixed $value): self
    {
        if (null === $value || '' === $value) {
            throw new EmptyPropertyValueException('Empty amount value', 0, 'amount.value');
        }
        if (!is_numeric($value)) {
            throw new InvalidPropertyValueTypeException('Invalid amount value type', 0, 'amount.value', $value);
        }
        if ((float) $value <= 0.0) {
            throw new InvalidPropertyValueException('Invalid amount value: "' . $value . '"', 0, 'amount.value', $value);
        }
        $castedValue = (int) round((float) $value * 100.0);
        if ($castedValue <= 0.0) {
            throw new InvalidPropertyValueException('Invalid amount value: "' . $value . '"', 0, 'amount.value', $value);
        }
        $this->_value = $this->validatePropertyValue('_value', $castedValue);
        return $this;
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
     * @param string $currency Код валюты
     *
     * @throws EmptyPropertyValueException Генерируется если было передано пустое значение
     * @throws InvalidPropertyValueTypeException Генерируется если было передано значение невалидного типа
     * @throws InvalidPropertyValueException Генерируется если был передан неподдерживаемый код валюты
     */
    public function setCurrency(string $currency): self
    {
        $this->_currency = $this->validatePropertyValue('_currency', $currency);
        return $this;
    }

    /**
     * Умножает текущую сумму на указанный коэффициент
     *
     * @param float $coefficient Множитель
     *
     * @throws EmptyPropertyValueException Выбрасывается если передано пустое значение
     * @throws InvalidPropertyValueTypeException Выбрасывается если было передано не число
     * @throws InvalidPropertyValueException Выбрасывается если переданное значение меньше или равно нулю, либо если
     *                                       после умножения получили значение равное нулю
     */
    public function multiply(float $coefficient): void
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
     * @param float $value Значение, которое будет прибавлено к текущему
     *
     * @throws EmptyPropertyValueException Выбрасывается если передано пустое значение
     * @throws InvalidPropertyValueTypeException Выбрасывается если было передано не число
     * @throws InvalidPropertyValueException Выбрасывается если после сложения получилась сумма меньше или равная нулю
     */
    public function increase(float $value): void
    {
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
