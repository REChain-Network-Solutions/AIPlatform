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

use DateTime;
use DateTimeZone;
use Exception;
use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Class OperationalDetails.
 *
 * Данные операционного реквизита чека
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $operationId Идентификатор операции (тег в 54 ФЗ — 1271)
 * @property string $operation_id Идентификатор операции (тег в 54 ФЗ — 1271)
 * @property DateTime $createdAt Время создания операции (тег в 54 ФЗ — 1273)
 * @property DateTime $created_at Время создания операции (тег в 54 ФЗ — 1273)
 * @property string $value Данные операции (тег в 54 ФЗ — 1272)
 */
class OperationalDetails extends AbstractObject
{
    /** @var int Минимальное значение */
    public const MIN_VALUE = 0;

    /** @var int Максимальная длинна номера операции */
    public const OPERATION_ID_MAX_VALUE = 255;

    /** @var int Максимальная длинна значение операционного реквизита */
    public const VALUE_MAX_LENGTH = 64;

    /**
     * @var int|null Идентификатор операции (тег в 54 ФЗ — 1271). Число от 0 до 255
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(self::MIN_VALUE)]
    #[Assert\LessThanOrEqual(self::OPERATION_ID_MAX_VALUE)]
    private ?int $_operation_id = null;

    /**
     * @var DateTime|null Время создания операции (тег в 54 ФЗ — 1273).
     *               Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
     */
    #[Assert\NotBlank]
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_created_at = null;

    /**
     * @var string|null Данные операции (тег в 54 ФЗ — 1272)
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::VALUE_MAX_LENGTH)]
    private ?string $_value = null;

    /**
     * Возвращает идентификатор операции.
     *
     * @return int|null Идентификатор операции
     */
    public function getOperationId(): ?int
    {
        return $this->_operation_id;
    }

    /**
     * Устанавливает идентификатор операции.
     *
     * @param int|null $operation_id Идентификатор операции
     *
     * @return self
     */
    public function setOperationId(?int $operation_id = null): self
    {
        $this->_operation_id = $this->validatePropertyValue('_operation_id', $operation_id);
        return $this;
    }

    /**
     * Возвращает время создания операции.
     *
     * @return DateTime|null Время создания операции
     */
    public function getCreatedAt(): ?DateTime
    {
        return $this->_created_at;
    }

    /**
     * Устанавливает время создания операции.
     *
     * @param DateTime|string|null $created_at Время создания операции
     *
     * @return self
     *
     * @throws Exception
     */
    public function setCreatedAt(DateTime|string|null $created_at = null): self
    {
        $this->_created_at = $this->validatePropertyValue('_created_at', $created_at);
        return $this;
    }

    /**
     * Возвращает данные операции.
     *
     * @return string|null Данные операции
     */
    public function getValue(): ?string
    {
        return $this->_value;
    }

    /**
     * Устанавливает данные операции.
     *
     * @param string|null $value Данные операции
     *
     * @return self
     */
    public function setValue(?string $value = null): self
    {
        $this->_value = $this->validatePropertyValue('_value', $value);
        return $this;
    }

    /**
     * {@inheritdoc}
     */
    public function jsonSerialize(): array
    {
        $result = parent::jsonSerialize();
        $result['created_at'] = $this->getCreatedAt()
            ->setTimezone(new DateTimeZone('UTC'))
            ->format(YOOKASSA_DATE)
        ;

        return $result;
    }
}
